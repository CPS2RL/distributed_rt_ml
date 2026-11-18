
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "nvs_flash.h"
#include "model_adapter.h"
#include "partition_config.h"

#if EXP_DATASET_ID == 1
#if __has_include("mnist_image_generated.h")
#include "mnist_image_generated.h"
#define REAL_IMAGE 1
#else
#define REAL_IMAGE 0
#endif
#else
#if __has_include("cifar10_image_generated.h")
#include "cifar10_image_generated.h"
#define REAL_IMAGE 1
#else
#define REAL_IMAGE 0
#endif
#endif

#define IMAGE_W     EXP_IMG_W
#define IMAGE_H     EXP_IMG_W
#define IMAGE_C     EXP_IMG_C
#define ROW_BYTES   (IMAGE_W * IMAGE_C)
#define MAX_ROWS    32
#define CHANNEL     1
#define MAGIC       0xAB71
#define VERSION     6
#define TYPE_SETUP       1
#define TYPE_SETUP_ACK   2
#define TYPE_DATA        3
#define TYPE_DATA_ACK    4
#define TYPE_START       5
#define TYPE_TIMES       6
#define TYPE_TIMES_ACK   7
#define CONFIG_TAG   ((EXP_MODEL_ID << 4) | EXP_DATASET_ID)
#define RETRIES 35
#define MAX_RUN_WAIT_MS 180000
#if EXP_DATASET_ID == 1
#define CHUNK_ROWS 4
#else
#define CHUNK_ROWS 2
#endif
/* Use 192 bytes for both RGB row chunks and up to 32 layer timings. */
#define PAYLOAD_BYTES 192

/* The coordinator sends expanded, overlapping input intervals.
 * Derived in partition_config.h from final output ownership. */
#define ROW_START_ID(id) redundant_input_span(id).lo
#define ROW_END_ID(id) redundant_input_span(id).hi

static const uint8_t MACS[3][6] = {
    {0x28, 0x05, 0xA5, 0x74, 0x04, 0xE0},   /* COM3 */
    {0x28, 0x05, 0xA5, 0x73, 0xFA, 0xE0},   /* COM7 */
    {0x28, 0x05, 0xA5, 0x74, 0x52, 0xD4}    /* COM8 */
};

typedef struct __attribute__((packed)) {
    uint16_t magic;
    uint8_t version;
    uint8_t config_tag;
    uint8_t type;
    uint8_t sender_id;
    uint8_t run_id;
    uint8_t first_row;
    uint8_t row_count;
    uint8_t flags;
    uint32_t hash;
    uint8_t payload[PAYLOAD_BYTES];
} packet_t;
#define HEADER_SIZE offsetof(packet_t, payload)
_Static_assert(sizeof(packet_t) <= 250, "ESP-NOW v1 packet exceeds 250 bytes");
_Static_assert(MAX_TIMED_LAYERS * sizeof(uint32_t) <= PAYLOAD_BYTES,
               "Timing result too long for packet payload");
_Static_assert(CHUNK_ROWS * ROW_BYTES <= PAYLOAD_BYTES,
               "Input chunk exceeds radio payload");

typedef struct {
    uint8_t source_mac[6];
    uint16_t len;
    packet_t p;
} rx_event_t;

static QueueHandle_t s_queue;
static int s_id;
static uint8_t s_full_image[IMAGE_W * IMAGE_H * IMAGE_C];
static uint8_t s_local[MAX_ROWS * ROW_BYTES];
static bool s_worker_ready;
static uint32_t s_rx_mask;
static uint8_t s_worker_last_run;
static bool s_pending;
static packet_t s_pending_packet;

static uint32_t hash32(const uint8_t *buf, size_t size)
{
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < size; ++i) hash = (hash ^ buf[i]) * 16777619u;
    return hash;
}
static size_t local_bytes(int id)
{
    return (size_t)(ROW_END_ID(id) - ROW_START_ID(id)) * ROW_BYTES;
}
static int recognize_device(void)
{
    uint8_t mac[6];
    ESP_ERROR_CHECK(esp_read_mac(mac, ESP_MAC_WIFI_STA));
    for (int id = 0; id < 3; ++id)
        if (memcmp(mac, MACS[id], 6) == 0) return id;
    return -1;
}
static void send_packet(const uint8_t dest[6], const packet_t *p, size_t length)
{
    (void)esp_now_send(dest, (const uint8_t *)p, length);
}
static packet_t new_packet(uint8_t type, uint8_t id)
{
    packet_t p = {0};
    p.magic = MAGIC;
    p.version = VERSION;
    p.config_tag = CONFIG_TAG;
    p.type = type;
    p.sender_id = id;
    return p;
}
static void callback_recv(const esp_now_recv_info_t *info, const uint8_t *data, int size)
{
    if (!info || !info->src_addr || !data || size < (int)HEADER_SIZE ||
        size > (int)sizeof(packet_t)) return;
    rx_event_t ev = {0};
    memcpy(ev.source_mac, info->src_addr, 6);
    ev.len = (uint16_t)size;
    memcpy(&ev.p, data, (size_t)size);
    if (ev.p.magic != MAGIC || ev.p.version != VERSION ||
        ev.p.config_tag != CONFIG_TAG) return;
    (void)xQueueSend(s_queue, &ev, 0);
}
static void radio_init(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_recv_cb(callback_recv));
}
static void add_peer(int id)
{
    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, MACS[id], 6);
    peer.channel = CHANNEL;
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
}
static void make_input(void)
{
#if REAL_IMAGE && EXP_DATASET_ID == 1
    memcpy(s_full_image, mnist_image_28x28, sizeof(s_full_image));
#elif REAL_IMAGE && EXP_DATASET_ID == 2
    memcpy(s_full_image, cifar10_image_32x32_rgb, sizeof(s_full_image));
#else
    /* Deterministic synthetic image: suitable for timing, not accuracy. */
    for (size_t i = 0; i < sizeof(s_full_image); ++i)
        s_full_image[i] = (uint8_t)((i * 13u + (i / ROW_BYTES) * 7u) & 255u);
#endif
}
static bool authentic_source(const rx_event_t *ev, int id)
{
    return id >= 0 && id < 3 && memcmp(ev->source_mac, MACS[id], 6) == 0;
}

/* Setup/data use stop-and-wait ACKs; image transfers are not timed. */
static bool send_and_confirm(int worker, const packet_t *p, size_t length,
                             uint8_t ack_type, uint32_t expected_hash)
{
    for (int attempt = 0; attempt < RETRIES; ++attempt) {
        send_packet(MACS[worker], p, length);
        TickType_t start = xTaskGetTickCount();
        rx_event_t ev;
        while (xTaskGetTickCount() - start < pdMS_TO_TICKS(600)) {
            if (xQueueReceive(s_queue, &ev, pdMS_TO_TICKS(100)) != pdTRUE) continue;
            const packet_t *a = &ev.p;
            if (!authentic_source(&ev, worker) || ev.len != HEADER_SIZE ||
                a->type != ack_type || a->sender_id != worker ||
                a->first_row != p->first_row || a->row_count != p->row_count)
                continue;
            if (ack_type == TYPE_DATA_ACK && p->flags == 1 && a->hash != expected_hash)
                return false;
            return true;
        }
    }
    return false;
}
static bool distribute_input(void)
{
    for (int worker = 1; worker <= 2; ++worker) {
        packet_t setup = new_packet(TYPE_SETUP, 0);
        if (!send_and_confirm(worker, &setup, HEADER_SIZE, TYPE_SETUP_ACK, 0))
            return false;
        const int begin = ROW_START_ID(worker);
        const int end = ROW_END_ID(worker);
        const uint32_t expected_hash = hash32(s_full_image + begin * ROW_BYTES,
                                               local_bytes(worker));
        for (int first = begin; first < end; first += CHUNK_ROWS) {
            int rows = end - first;
            if (rows > CHUNK_ROWS) rows = CHUNK_ROWS;
            packet_t pkt = new_packet(TYPE_DATA, 0);
            pkt.first_row = (uint8_t)first;
            pkt.row_count = (uint8_t)rows;
            pkt.flags = (first + rows == end) ? 1 : 0;
            memcpy(pkt.payload, s_full_image + first * ROW_BYTES, rows * ROW_BYTES);
            if (!send_and_confirm(worker, &pkt,
                                  HEADER_SIZE + rows * ROW_BYTES,
                                  TYPE_DATA_ACK, expected_hash)) return false;
        }
    }
    return true;
}

static void send_worker_ack(uint8_t ack_type, const packet_t *request,
                            uint32_t checksum)
{
    packet_t reply = new_packet(ack_type, (uint8_t)s_id);
    reply.run_id = request->run_id;
    reply.first_row = request->first_row;
    reply.row_count = request->row_count;
    reply.hash = checksum;
    send_packet(MACS[0], &reply, HEADER_SIZE);
}
static void send_worker_times(uint8_t run)
{
    if (!s_pending || s_pending_packet.run_id != run) return;
    send_packet(MACS[0], &s_pending_packet,
                HEADER_SIZE + (size_t)selected_model_layers() * sizeof(uint32_t));
}
static void do_worker_run(uint8_t run)
{
    uint32_t us[MAX_TIMED_LAYERS];
    if (selected_model_infer(s_local, local_bytes(s_id), s_id, us) != 0) {
        /* Coordinator will timeout, rather than receive invented zeros. */
        return;
    }
    s_pending_packet = new_packet(TYPE_TIMES, (uint8_t)s_id);
    s_pending_packet.run_id = run;
    const size_t len = (size_t)selected_model_layers() * sizeof(uint32_t);
    memcpy(s_pending_packet.payload, us, len);
    s_pending_packet.hash = hash32(s_pending_packet.payload, len);
    s_pending = true;
    send_worker_times(run);
}
static void worker_task(void)
{
    rx_event_t ev;
    s_worker_ready = false;
    s_worker_last_run = 0;
    s_rx_mask = 0;
    s_pending = false;
    TickType_t last_retry = xTaskGetTickCount();
    const int first_local = ROW_START_ID(s_id);
    const int rows_local = ROW_END_ID(s_id) - first_local;
    const uint32_t complete_mask = rows_local == 32 ? UINT32_MAX :
                                   ((1u << rows_local) - 1u);
    for (;;) {
        if (xQueueReceive(s_queue, &ev, pdMS_TO_TICKS(150)) != pdTRUE) {
            if (s_pending && (xTaskGetTickCount() - last_retry >= pdMS_TO_TICKS(500))) {
                send_worker_times(s_pending_packet.run_id);
                last_retry = xTaskGetTickCount();
            }
            continue;
        }
        const packet_t *p = &ev.p;
        if (!authentic_source(&ev, 0) || p->sender_id != 0) continue;
        if (p->type == TYPE_SETUP && ev.len == HEADER_SIZE) {
            memset(s_local, 0, sizeof(s_local));
            s_worker_ready = false;
            s_worker_last_run = 0;
            s_rx_mask = 0;
            s_pending = false;
            send_worker_ack(TYPE_SETUP_ACK, p, 0);
            continue;
        }
        if (p->type == TYPE_DATA && !s_worker_ready &&
            p->row_count >= 1 && p->row_count <= CHUNK_ROWS &&
            p->first_row >= first_local &&
            p->first_row + p->row_count <= ROW_END_ID(s_id) &&
            ev.len == HEADER_SIZE + (size_t)p->row_count * ROW_BYTES) {
            for (int j = 0; j < p->row_count; ++j) {
                const int localrow = p->first_row + j - first_local;
                memcpy(s_local + (size_t)localrow * ROW_BYTES,
                       p->payload + (size_t)j * ROW_BYTES, ROW_BYTES);
                s_rx_mask |= 1u << localrow;
            }
            if (s_rx_mask == complete_mask) s_worker_ready = true;
            uint32_t checksum = s_worker_ready ? hash32(s_local, local_bytes(s_id)) : 0;
            send_worker_ack(TYPE_DATA_ACK, p, checksum);
            continue;
        }
        /* Respond to retransmission of the last data packet after its ACK was lost. */
        if (p->type == TYPE_DATA && s_worker_ready &&
            p->flags == 1 && p->first_row + p->row_count == ROW_END_ID(s_id)) {
            send_worker_ack(TYPE_DATA_ACK, p, hash32(s_local, local_bytes(s_id)));
            continue;
        }
        if (p->type == TYPE_TIMES_ACK && ev.len == HEADER_SIZE &&
            s_pending && p->run_id == s_pending_packet.run_id) {
            s_pending = false;
            continue;
        }
        if (p->type == TYPE_START && ev.len == HEADER_SIZE && s_worker_ready) {
            if (p->run_id == s_worker_last_run && s_pending) {
                send_worker_times(p->run_id);  /* duplicate START */
            } else if (p->run_id == (uint8_t)(s_worker_last_run + 1) && !s_pending) {
                s_worker_last_run = p->run_id;
                do_worker_run(p->run_id);
                last_retry = xTaskGetTickCount();
            }
        }
    }
}

static void send_start(uint8_t run)
{
    packet_t start = new_packet(TYPE_START, 0);
    start.run_id = run;
    send_packet(MACS[1], &start, HEADER_SIZE);
    send_packet(MACS[2], &start, HEADER_SIZE);
}
static bool receive_timings(uint8_t run, uint32_t times[3][MAX_TIMED_LAYERS])
{
    bool received[3] = {true, false, false};
    const size_t result_bytes = (size_t)selected_model_layers() * sizeof(uint32_t);
    const TickType_t begun = xTaskGetTickCount();
    TickType_t resent = 0;
    rx_event_t ev;
    while (!received[1] || !received[2]) {
        TickType_t now = xTaskGetTickCount();
        if (now - begun > pdMS_TO_TICKS(MAX_RUN_WAIT_MS)) return false;
        if (resent == 0 || now - resent > pdMS_TO_TICKS(1200)) {
            send_start(run);
            resent = xTaskGetTickCount();
        }
        if (xQueueReceive(s_queue, &ev, pdMS_TO_TICKS(200)) != pdTRUE) continue;
        const packet_t *p = &ev.p;
        int id = p->sender_id;
        if (p->type != TYPE_TIMES || id < 1 || id > 2 ||
            !authentic_source(&ev, id) || ev.len != HEADER_SIZE + result_bytes ||
            p->hash != hash32(p->payload, result_bytes)) continue;
        /* ACK even if this is a duplicate of an earlier run. */
        packet_t ack = new_packet(TYPE_TIMES_ACK, 0);
        ack.run_id = p->run_id;
        send_packet(MACS[id], &ack, HEADER_SIZE);
        if (p->run_id != run) continue;
        if (!received[id]) {
            memcpy(times[id], p->payload, result_bytes);
            received[id] = true;
        }
    }
    return true;
}
static void print_maximum_per_layer(int run, const uint32_t per_device[3][MAX_TIMED_LAYERS])
{
    const int n = selected_model_layers();
    printf("run %d\n", run);
    printf("execution time of layers (ms): {");
    for (int l = 0; l < n; ++l) {
        uint32_t maximum = per_device[0][l];
        for (int i = 1; i < 3; ++i)
            if (per_device[i][l] > maximum) maximum = per_device[i][l];
        printf("%s%s: %.3f", l ? ", " : "", selected_model_layer_name(l),
               (double)maximum / 1000.0);
    }
    printf("}\n");
    fflush(stdout);
}
static void coordinator_task(void)
{
    make_input();
    memcpy(s_local, s_full_image + (size_t)ROW_START_ID(0) * ROW_BYTES, local_bytes(0));
    if (!distribute_input()) {
        fprintf(stderr, "ERROR: input transfer failed; verify COM  and selected model/dataset.\n");
        return;
    }
    uint32_t layer_times[3][MAX_TIMED_LAYERS] = {{0}};
    for (int run = 1; run <= EXP_RUNS; ++run) {
        /* Workers start first. No per-layer synchronization or halo transfer. */
        send_start((uint8_t)run);
        if (selected_model_infer(s_local, local_bytes(0), 0, layer_times[0]) != 0) {
            fprintf(stderr, "ERROR: coordinator model inference failed on run %d.\n", run);
            return;
        }
        if (!receive_timings((uint8_t)run, layer_times)) {
            fprintf(stderr, "ERROR: missing worker layer times on run %d.\n", run);
            return;
        }
        print_maximum_per_layer(run, layer_times);
    }
}
void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_NONE);  
    s_id = recognize_device();
    if (s_id < 0) {
        fprintf(stderr, "ERROR: unassigned ESP32 MAC.\n");
        return;
    }
    if (!all_partitions_valid()) {
        fprintf(stderr, "ERROR: invalid layer boundaries in partition_config.h.\n");
        return;
    }
    if (!selected_model_ready()) {
        fprintf(stderr, "ERROR: trained weights not found for selected model/dataset.\n");
        return;
    }
    s_queue = xQueueCreate(24, sizeof(rx_event_t));
    if (!s_queue) {
        fprintf(stderr, "ERROR: insufficient memory for radio queue.\n");
        return;
    }
    radio_init();
    if (s_id == 0) {
        add_peer(1);
        add_peer(2);
        vTaskDelay(pdMS_TO_TICKS(1200));
        coordinator_task();
    } else {
        add_peer(0);
        worker_task();
    }
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
