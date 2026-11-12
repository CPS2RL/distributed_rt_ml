/* Three-ESP32 layer execution benchmark, all six CNN/dataset choices.
 * Shared firmware on each board, WITH intermediate FP32 halo exchange.
 * COM3 coordinator displays ONLY per-run per-layer maxima in milliseconds.
 * Times are local computation times (not Wi-Fi/ESP-NOW time).
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include "esp_timer.h"
#include "partition_config.h"
#include "halo_exchange.h"

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
#define VERSION     4
#define TYPE_SETUP       1
#define TYPE_SETUP_ACK   2
#define TYPE_DATA        3
#define TYPE_DATA_ACK    4
#define TYPE_START       5
#define TYPE_TIMES       6
#define TYPE_TIMES_ACK   7
#define TYPE_HREADY      8
#define TYPE_HREADY_ACK  9
#define TYPE_HGET       10
#define TYPE_HFRAGMENT  11
#define TYPE_HSET       12
#define TYPE_HSET_ACK   13
#define TYPE_HDONE      14
#define TYPE_HDONE_ACK  15
#define CONFIG_TAG   ((EXP_MODEL_ID << 4) | EXP_DATASET_ID)
#define RETRIES 35
#define MAX_RUN_WAIT_MS 900000
#if EXP_DATASET_ID == 1
#define CHUNK_ROWS 4
#else
#define CHUNK_ROWS 2
#endif
/* Use 192 bytes for both RGB row chunks and up to 32 layer timings. */
#define PAYLOAD_BYTES 192

/* All physical and model layer boundaries live in partition_config.h. */
#define ROW_START PART_SELECTED_INPUT
#define LOCAL_ROW_BEGIN(id) (PART_SELECTED_INPUT[(id)])
#define LOCAL_ROW_END(id) (PART_SELECTED_INPUT[(id)+1])

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
    return (size_t)(LOCAL_ROW_END(id) - LOCAL_ROW_BEGIN(id)) * ROW_BYTES;
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
        const int begin = LOCAL_ROW_BEGIN(worker);
        const int end = LOCAL_ROW_END(worker);
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


/* --------------------- INTERMEDIATE HALO EXCHANGE -----------------------
 * All three nodes call halo_prepare() at identical operation steps. Workers
 * signal readiness. COM3 then fetches ONLY source rows required by another
 * output stripe and forwards the exact FP32 data to the consuming worker.
 * GET/FRAGMENT and SET/ACK are stop-and-wait and chunk each row <=192 bytes.
 * Communication/barrier time is kept separate from compute layer timings.
 */
#define HALO_H_MAX 32
#define HALO_RADIO_ATTEMPTS 100
static uint8_t s_halo_run;
static uint32_t s_halo_us[MAX_TIMED_LAYERS];
static int s_last_halo_done = -1;
static float *s_halo_remote = NULL;
static size_t s_halo_remote_cap = 0;
static int s_halo_row_to_index[HALO_H_MAX];
static uint8_t s_halo_need[3][HALO_H_MAX];
static size_t s_halo_row_bytes;
static int s_halo_remote_count;
static const HaloSpec *s_halo_spec;
static float *s_halo_cache = NULL;
static size_t s_halo_cache_cap = 0;
static int s_halo_cache_map[HALO_H_MAX];

static int halo_owner(const uint8_t *bounds, int row)
{
    for (int i=0; i<3; i++)
        if (row >= bounds[i] && row < bounds[i+1]) return i;
    return -1;
}
static int halo_peer_message(const rx_event_t *ev, int src, uint8_t type,
                             int step, int row, int fragment)
{
    const packet_t *p = &ev->p;
    return authentic_source(ev, src) && p->sender_id == src &&
           p->run_id == s_halo_run && p->flags == (uint8_t)step &&
           p->type == type &&
           (row < 0 || p->first_row == (uint8_t)row) &&
           (fragment < 0 || p->row_count == (uint8_t)fragment);
}
static packet_t halo_packet(uint8_t type, int step, int row, int frag)
{
    packet_t p = new_packet(type, (uint8_t)s_id);
    p.flags = (uint8_t)step;
    p.run_id = s_halo_run;
    p.first_row = (uint8_t)row;
    p.row_count = (uint8_t)frag;
    return p;
}
static void halo_ack(const packet_t *p, int dest, uint8_t type)
{
    packet_t ack = halo_packet(type, p->flags, p->first_row, p->row_count);
    ack.hash = p->hash;
    send_packet(MACS[dest], &ack, HEADER_SIZE);
}
static void halo_ack_duplicates(const rx_event_t *ev, int current_step)
{
    const packet_t *p = &ev->p;
    if (s_id == 0) {
        int from = p->sender_id;
        if (from > 0 && from < 3 &&
            halo_peer_message(ev, from, TYPE_HREADY, current_step, -1, -1))
            halo_ack(p, from, TYPE_HREADY_ACK);
    } else if (authentic_source(ev, 0) && p->run_id == s_halo_run &&
               p->type == TYPE_HDONE && p->flags == (uint8_t)s_last_halo_done)
        halo_ack(p, 0, TYPE_HDONE_ACK);
}
static int halo_wait_response(int from, uint8_t type, int step, int row, int frag,
                              rx_event_t *found, int ms)
{
    const int64_t deadline = esp_timer_get_time() + (int64_t)ms * 1000;
    rx_event_t ev;
    while (esp_timer_get_time() < deadline) {
        if (xQueueReceive(s_queue, &ev, pdMS_TO_TICKS(40)) != pdTRUE) continue;
        if (halo_peer_message(&ev, from, type, step, row, frag)) {
            if (found) *found = ev;
            return 0;
        }
        halo_ack_duplicates(&ev, step);
    }
    return -1;
}
static int halo_send_and_ack(int dest, const packet_t *p, size_t nbytes,
                             uint8_t ack_type, int step)
{
    for (int tries=0; tries<HALO_RADIO_ATTEMPTS; tries++) {
        send_packet(MACS[dest], p, nbytes);
        if (halo_wait_response(dest, ack_type, step, p->first_row,
                               p->row_count, NULL, 300) == 0) return 0;
    }
    return -1;
}
static int halo_build_plan(const HaloSpec *s)
{
    if (!s || !s->src || !s->source_bounds || !s->destination_bounds ||
        s->step < 0 || s->step >= MAX_TIMED_LAYERS ||
        s->source_height < 1 || s->source_height > HALO_H_MAX ||
        s->width < 1 || s->channels < 1 || s->kernel < 1 ||
        s->stride < 1 || s->pad < 0) return -1;
    s_halo_spec = s;
    s_halo_row_bytes = (size_t)s->width * s->channels * sizeof(float);
    memset(s_halo_need, 0, sizeof(s_halo_need));
    for (int dev=0; dev<3; dev++) {
        const int first = s->destination_bounds[dev];
        const int end = s->destination_bounds[dev+1];
        if (end <= first) continue;
        int low = first * s->stride - s->pad;
        int high = (end-1) * s->stride - s->pad + s->kernel - 1;
        if (low < 0) low = 0;
        if (high >= s->source_height) high = s->source_height - 1;
        for (int row=low; row<=high; row++) {
            int owner=halo_owner(s->source_bounds, row);
            if (owner < 0) return -1;
            if (owner != dev) s_halo_need[dev][row] = 1;
        }
    }
    for (int row=0; row<HALO_H_MAX; row++) s_halo_row_to_index[row] = -1;
    s_halo_remote_count = 0;
    for (int row=0; row<s->source_height; row++)
        if (s_halo_need[s_id][row]) s_halo_row_to_index[row] = s_halo_remote_count++;
    size_t bytes = (size_t)s_halo_remote_count * s_halo_row_bytes;
    if (s->compact && bytes > s_halo_remote_cap) {
        float *ptr = realloc(s_halo_remote, bytes);
        if (!ptr) return -1;
        s_halo_remote = ptr;
        s_halo_remote_cap = bytes;
    }
    return 0;
}
static float *halo_owned_row(const HaloSpec *s, int row)
{
    int first = s->source_bounds[s_id];
    int last = s->source_bounds[s_id+1];
    if (row < first || row >= last) return NULL;
    return s->src + (size_t)(s->compact ? row-first : row) * s->width * s->channels;
}
static float *halo_remote_row(const HaloSpec *s, int row)
{
    if (row < 0 || row >= s->source_height) return NULL;
    int ri = s_halo_row_to_index[row];
    if (ri < 0) return NULL;
    if (!s->compact)
        return s->src + (size_t)row * s->width * s->channels;
    return s_halo_remote + (size_t)ri * s->width * s->channels;
}
const float *halo_row(const float *local, int global_row,
                      int first, int last, int width, int channels)
{
    if (global_row >= first && global_row < last)
        return local + (size_t)(global_row-first) * width * channels;
    if (!s_halo_spec || s_halo_spec->width != width ||
        s_halo_spec->channels != channels) return NULL;
    return halo_remote_row(s_halo_spec, global_row);
}
static int halo_worker(const HaloSpec *s)
{
    const int step=s->step;
    packet_t ready=halo_packet(TYPE_HREADY,step,0,0);
    if (halo_send_and_ack(0,&ready,HEADER_SIZE,TYPE_HREADY_ACK,step) != 0)
        return -1;
    /* Worker services coordinator GET/SET commands until it receives DONE. */
    for (;;) {
        rx_event_t ev;
        if (xQueueReceive(s_queue,&ev,pdMS_TO_TICKS(200)) != pdTRUE) continue;
        packet_t *p=&ev.p;
        if (!authentic_source(&ev,0) || p->sender_id!=0 ||
            p->run_id!=s_halo_run) continue;
        if (p->flags==(uint8_t)step && p->type==TYPE_HGET && ev.len==HEADER_SIZE) {
            int row=p->first_row, frag=p->row_count;
            float *source=halo_owned_row(s,row);
            size_t offset=(size_t)frag*PAYLOAD_BYTES;
            if (!source || offset>=s_halo_row_bytes) continue;
            size_t n=s_halo_row_bytes-offset;
            if (n>PAYLOAD_BYTES) n=PAYLOAD_BYTES;
            packet_t reply=halo_packet(TYPE_HFRAGMENT,step,row,frag);
            memcpy(reply.payload,(const uint8_t*)source+offset,n);
            reply.hash=hash32(reply.payload,n);
            send_packet(MACS[0],&reply,HEADER_SIZE+n);
        } else if (p->flags==(uint8_t)step && p->type==TYPE_HSET) {
            int row=p->first_row, frag=p->row_count;
            size_t offset=(size_t)frag*PAYLOAD_BYTES;
            size_t n=s_halo_row_bytes>offset?s_halo_row_bytes-offset:0;
            if (n>PAYLOAD_BYTES) n=PAYLOAD_BYTES;
            float *dest=halo_remote_row(s,row);
            if (!dest || n==0 || ev.len != HEADER_SIZE+n ||
                hash32(p->payload,n)!=p->hash) continue;
            memcpy((uint8_t*)dest+offset,p->payload,n);
            halo_ack(p,0,TYPE_HSET_ACK);
        } else if (p->flags==(uint8_t)step && p->type==TYPE_HDONE &&
                   ev.len==HEADER_SIZE) {
            halo_ack(p,0,TYPE_HDONE_ACK);
            s_last_halo_done=step;
            return 0;
        } else {
            halo_ack_duplicates(&ev,step);
        }
    }
}
static int halo_coordinator(const HaloSpec *s)
{
    int step=s->step;
    int ready[3]={1,0,0};
    int64_t deadline=esp_timer_get_time()+20000000; /* 20s per barrier */
    while ((!ready[1] || !ready[2]) && esp_timer_get_time()<deadline) {
        rx_event_t ev;
        if (xQueueReceive(s_queue,&ev,pdMS_TO_TICKS(100)) != pdTRUE) continue;
        int from=ev.p.sender_id;
        if (from<1 || from>2) continue;
        if (halo_peer_message(&ev,from,TYPE_HREADY,step,-1,-1) &&
            ev.len==HEADER_SIZE) {
            halo_ack(&ev.p,from,TYPE_HREADY_ACK);
            ready[from]=1;
        } else halo_ack_duplicates(&ev,step);
    }
    if (!ready[1] || !ready[2]) return -1;
    /* Cache required boundary rows from workers. No bulk feature-map exchange. */
    for (int row=0; row<HALO_H_MAX;row++) s_halo_cache_map[row]=-1;
    int cache_rows=0;
    for (int row=0;row<s->source_height;row++) {
        int owner=halo_owner(s->source_bounds,row);
        if (owner<1) continue; /* coordinator already has its own rows */
        for (int dev=0;dev<3;dev++) if (dev!=owner && s_halo_need[dev][row]) {
            s_halo_cache_map[row]=cache_rows++;
            break;
        }
    }
    size_t bytes=(size_t)cache_rows*s_halo_row_bytes;
    if (bytes>s_halo_cache_cap) {
        float *newcache=realloc(s_halo_cache,bytes);
        if (!newcache) return -1;
        s_halo_cache=newcache;
        s_halo_cache_cap=bytes;
    }
    for (int row=0;row<s->source_height;row++) {
        if (s_halo_cache_map[row]<0) continue;
        int owner=halo_owner(s->source_bounds,row);
        uint8_t *dest=(uint8_t*)(s_halo_cache +
                         (size_t)s_halo_cache_map[row]*s->width*s->channels);
        int frags=(s_halo_row_bytes+PAYLOAD_BYTES-1)/PAYLOAD_BYTES;
        for (int f=0;f<frags;f++) {
            packet_t get=halo_packet(TYPE_HGET,step,row,f);
            int got=0;
            size_t offset=(size_t)f*PAYLOAD_BYTES;
            size_t n=s_halo_row_bytes-offset;
            if(n>PAYLOAD_BYTES)n=PAYLOAD_BYTES;
            for (int tries=0;tries<HALO_RADIO_ATTEMPTS;tries++) {
                send_packet(MACS[owner],&get,HEADER_SIZE);
                rx_event_t reply;
                if (halo_wait_response(owner,TYPE_HFRAGMENT,step,row,f,&reply,300)==0 &&
                    reply.len==HEADER_SIZE+n &&
                    reply.p.hash==hash32(reply.p.payload,n)) {
                    memcpy(dest+offset,reply.p.payload,n);
                    got=1;
                    break;
                }
            }
            if (!got) return -1;
        }
    }
    /* Fill coordinator's own remote read windows. */
    for (int row=0;row<s->source_height;row++) {
        if (!s_halo_need[0][row]) continue;
        float *dest=halo_remote_row(s,row);
        if (!dest || s_halo_cache_map[row]<0) return -1;
        memcpy(dest,s_halo_cache+(size_t)s_halo_cache_map[row]*s->width*s->channels,
               s_halo_row_bytes);
    }
    /* Forward required rows to workers, not complete activations. */
    for (int worker=1;worker<=2;worker++) {
        for (int row=0;row<s->source_height;row++) {
            if (!s_halo_need[worker][row]) continue;
            int owner=halo_owner(s->source_bounds,row);
            const float *srcrow=NULL;
            if (owner==0) srcrow=halo_owned_row(s,row);
            else if (s_halo_cache_map[row]>=0)
                srcrow=s_halo_cache+(size_t)s_halo_cache_map[row]*s->width*s->channels;
            if (!srcrow) return -1;
            int frags=(s_halo_row_bytes+PAYLOAD_BYTES-1)/PAYLOAD_BYTES;
            for (int f=0;f<frags;f++) {
                packet_t set=halo_packet(TYPE_HSET,step,row,f);
                size_t offset=(size_t)f*PAYLOAD_BYTES;
                size_t n=s_halo_row_bytes-offset;
                if(n>PAYLOAD_BYTES)n=PAYLOAD_BYTES;
                memcpy(set.payload,(const uint8_t*)srcrow+offset,n);
                set.hash=hash32(set.payload,n);
                if (halo_send_and_ack(worker,&set,HEADER_SIZE+n,
                                      TYPE_HSET_ACK,step)!=0) return -1;
            }
        }
    }
    for (int worker=1;worker<=2;worker++) {
        packet_t done=halo_packet(TYPE_HDONE,step,0,0);
        if (halo_send_and_ack(worker,&done,HEADER_SIZE,TYPE_HDONE_ACK,step)!=0)
            return -1;
    }
    return 0;
}
void halo_begin_run(int run)
{
    s_halo_run=(uint8_t)run;
    s_last_halo_done=-1;
    s_halo_spec=NULL;
    memset(s_halo_us,0,sizeof(s_halo_us));
}
uint32_t halo_step_us(int step)
{
    return step>=0 && step<MAX_TIMED_LAYERS?s_halo_us[step]:0;
}
int halo_prepare(const HaloSpec *s)
{
    if (halo_build_plan(s)!=0) return -1;
    int64_t begun=esp_timer_get_time();
    int result=s_id==0?halo_coordinator(s):halo_worker(s);
    s_halo_us[s->step]=(uint32_t)(esp_timer_get_time()-begun);
    return result;
}
/* ------------------- END OF HALO TRANSPORT ------------------------------ */

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
    halo_begin_run((int)run);
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
    const int first_local = LOCAL_ROW_BEGIN(s_id);
    const int rows_local = LOCAL_ROW_END(s_id) - first_local;
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
            p->first_row + p->row_count <= LOCAL_ROW_END(s_id) &&
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
            p->flags == 1 && p->first_row + p->row_count == LOCAL_ROW_END(s_id)) {
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
    memcpy(s_local, s_full_image, local_bytes(0));
    if (!distribute_input()) {
        fprintf(stderr, "ERROR: ESP-NOW input transfer failed; verify COM7/COM8 and selected model/dataset.\n");
        return;
    }
    uint32_t layer_times[3][MAX_TIMED_LAYERS] = {{0}};
    for (int run = 1; run <= EXP_RUNS; ++run) {
        /* The three workers execute lockstep halo synchronization inside each model. */
        halo_begin_run(run);
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
#if EXP_REPORT_HALO
        printf("halo exchange time of layers (ms): {");
        for (int layer = 0; layer < selected_model_layers(); ++layer) {
            printf("%s%s: %.3f", layer ? ", " : "",
                   selected_model_layer_name(layer),
                   (double)halo_step_us(layer) / 1000.0);
        }
        printf("}\n");
        fflush(stdout);
#endif
    }
}
void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_NONE);   /* suppress application Wi-Fi logging */
    s_id = recognize_device();
    if (s_id < 0) {
        fprintf(stderr, "ERROR: unassigned ESP32 MAC.\n");
        return;
    }
    if (!all_partitions_valid()) {
        fprintf(stderr, "ERROR: invalid partition_config.h boundaries.\n");
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
