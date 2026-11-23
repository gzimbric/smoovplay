// See frametap.h.
#include "frametap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#define TOTAL_SIZE ((size_t)FRAMETAP_HEADER_SIZE + (size_t)FRAMETAP_SLOT_COUNT * FRAMETAP_SLOT_SIZE)
#define AUDIO_TOTAL_SIZE ((size_t)FRAMETAP_AUDIO_HEADER + (size_t)FRAMETAP_AUDIO_SLOT_COUNT * FRAMETAP_AUDIO_SLOT_SIZE)

static HANDLE g_map = NULL;
static HANDLE g_event = NULL;
static uint8_t *g_base = NULL;
static HANDLE g_amap = NULL;
static uint8_t *g_abase = NULL; // the sound ring; NULL if it couldn't be made (the frames still flow)
static uint32_t g_arate = 48000, g_achannels = 2;
static int g_enabled = -1;

#define H32(off) ((volatile uint32_t *)(g_base + (off)))
#define H64(off) ((volatile uint64_t *)(g_base + (off)))
#define A32(off) ((volatile uint32_t *)(g_abase + (off)))
#define A64(off) ((volatile uint64_t *)(g_abase + (off)))

bool frametap_enabled(void)
{
	if(g_enabled < 0)
	{
		const char *v = getenv("CHIAKI_FRAMETAP");
		g_enabled = (v && v[0] == '1') ? 1 : 0;
	}
	return g_enabled == 1;
}

static bool open_shared(void)
{
	if(g_base)
		return true;
	const char *name = getenv("CHIAKI_FRAMETAP_NAME");
	if(!name || !name[0] || strlen(name) > 200)
		name = FRAMETAP_NAME;
	char event_name[256];
	snprintf(event_name, sizeof(event_name), "%s-frame", name);
	// Either side may create the mapping first; CreateFileMapping opens the existing one if it's there.
	g_map = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
		(DWORD)((uint64_t)TOTAL_SIZE >> 32), (DWORD)(TOTAL_SIZE & 0xffffffffu), name);
	if(!g_map)
		return false;
	g_base = (uint8_t *)MapViewOfFile(g_map, FILE_MAP_ALL_ACCESS, 0, 0, TOTAL_SIZE);
	if(!g_base)
	{
		CloseHandle(g_map);
		g_map = NULL;
		return false;
	}
	g_event = CreateEventA(NULL, FALSE, FALSE, event_name); // auto-reset, one reader

	char audio_name[256];
	snprintf(audio_name, sizeof(audio_name), "%s" FRAMETAP_AUDIO_SUFFIX, name);
	g_amap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)AUDIO_TOTAL_SIZE, audio_name);
	if(g_amap)
	{
		g_abase = (uint8_t *)MapViewOfFile(g_amap, FILE_MAP_ALL_ACCESS, 0, 0, AUDIO_TOTAL_SIZE);
		if(!g_abase)
		{
			CloseHandle(g_amap);
			g_amap = NULL;
		}
	}
	return true;
}

static void audio_header(uint64_t qpc_freq)
{
	if(!g_abase)
		return;
	InterlockedExchange((volatile LONG *)A32(0), 0);
	*A32(4) = 1;
	*A32(8) = FRAMETAP_AUDIO_SLOT_COUNT;
	*A32(12) = FRAMETAP_AUDIO_SLOT_SIZE;
	*A32(16) = g_arate;
	*A32(20) = g_achannels;
	*A64(32) = qpc_freq;
	InterlockedExchange64((volatile LONG64 *)A64(24), 0);
	InterlockedExchange((volatile LONG *)A32(0), (LONG)FRAMETAP_AUDIO_MAGIC);
}

bool frametap_begin(uint32_t codec, uint32_t width, uint32_t height, uint32_t max_fps)
{
	if(!frametap_enabled() || !open_shared())
		return false;
	InterlockedExchange((volatile LONG *)H32(0), 0); // header is being rewritten
	LARGE_INTEGER f;
	QueryPerformanceFrequency(&f);
	*H32(4) = FRAMETAP_VERSION;
	*H32(8) = FRAMETAP_SLOT_COUNT;
	*H32(12) = FRAMETAP_SLOT_SIZE;
	*H32(16) = codec;
	*H32(20) = width;
	*H32(24) = height;
	*H32(28) = max_fps;
	*H64(48) = (uint64_t)f.QuadPart;
	*H32(40) = 0;
	*H32(64) = FRAMETAP_FEATURE_PRESS | FRAMETAP_FEATURE_AUDIO_DELAY | (g_abase ? FRAMETAP_FEATURE_AUDIO_TAP : 0u);
	*H32(68) = 0;
	audio_header((uint64_t)f.QuadPart);
	// The reader treats a new session as a new stream: it resets its decoder and waits for a keyframe.
	InterlockedExchange64((volatile LONG64 *)H64(32), 0);
	InterlockedIncrement((volatile LONG *)H32(44));
	InterlockedExchange((volatile LONG *)H32(0), (LONG)FRAMETAP_MAGIC); // last: header is complete
	return true;
}

void frametap_push(const uint8_t *buf, size_t size, int32_t frames_lost, bool recovered)
{
	if(!g_base || *H32(0) != FRAMETAP_MAGIC)
		return;
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	uint64_t seq = *H64(32);
	uint8_t *slot = g_base + FRAMETAP_HEADER_SIZE + (size_t)(seq % FRAMETAP_SLOT_COUNT) * FRAMETAP_SLOT_SIZE;
	size_t cap = FRAMETAP_SLOT_SIZE - FRAMETAP_SLOT_HEADER;
	uint32_t flags = 0;
	if(size > cap)
	{
		size = cap;
		flags |= 1;
	}
	// Mark the slot as being rewritten first, so a reader copying it sees the sequence change and retries.
	*(volatile uint64_t *)(slot + 0) = UINT64_MAX;
	MemoryBarrier();
	*(volatile uint64_t *)(slot + 8) = (uint64_t)now.QuadPart;
	*(volatile uint32_t *)(slot + 16) = (uint32_t)size;
	*(volatile int32_t *)(slot + 20) = frames_lost;
	*(volatile uint32_t *)(slot + 24) = recovered ? 1u : 0u;
	*(volatile uint32_t *)(slot + 28) = flags;
	memcpy(slot + FRAMETAP_SLOT_HEADER, buf, size);
	MemoryBarrier();
	*(volatile uint64_t *)(slot + 0) = seq;
	InterlockedExchange64((volatile LONG64 *)H64(32), (LONG64)(seq + 1));
	if(g_event)
		SetEvent(g_event);
}

bool frametap_take_idr_request(void)
{
	if(!g_base)
		return false;
	return InterlockedCompareExchange((volatile LONG *)H32(40), 0, 1) == 1;
}

uint32_t frametap_take_press_request(uint32_t *hold_ms)
{
	if(!g_base)
		return 0;
	uint32_t buttons = (uint32_t)InterlockedExchange((volatile LONG *)H32(68), 0);
	if(buttons && hold_ms)
	{
		uint32_t h = *H32(72);
		*hold_ms = h < 20 ? 20 : h > 2000 ? 2000 : h;
	}
	return buttons;
}

void frametap_press_sent(void)
{
	if(!g_base)
		return;
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	InterlockedExchange64((volatile LONG64 *)H64(80), (LONG64)now.QuadPart);
	InterlockedIncrement((volatile LONG *)H32(76));
}

uint32_t frametap_audio_delay_ms(void)
{
	if(!g_base || *H32(0) != FRAMETAP_MAGIC)
		return 0;
	uint32_t ms = *H32(88);
	uint64_t freq = *H64(48);
	if(!ms || !freq)
		return 0;
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	if((int64_t)((uint64_t)now.QuadPart - *H64(56)) > (int64_t)(2 * freq)) // SmoovPlay stopped
		return 0;
	return ms > 500 ? 500 : ms;
}

void frametap_audio_format(uint32_t rate, uint32_t channels)
{
	if(!rate || !channels || channels > 8)
		return;
	g_arate = rate;
	g_achannels = channels;
	if(g_abase && *A32(0) == FRAMETAP_AUDIO_MAGIC)
	{
		*A32(16) = rate;
		*A32(20) = channels;
	}
}

void frametap_push_audio(const uint8_t *buf, size_t size)
{
	if(!g_abase || *A32(0) != FRAMETAP_AUDIO_MAGIC || !buf || !size)
		return;
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	uint64_t freq = *A64(32);
	size_t frame_bytes = 2u * g_achannels;
	size_t cap = (FRAMETAP_AUDIO_SLOT_SIZE - FRAMETAP_AUDIO_SLOT_HEADER) / frame_bytes * frame_bytes;
	size_t done = 0;
	while(done < size)
	{
		size_t n = size - done < cap ? size - done : cap;
		// A chunk too big for one slot goes in several; each slot's time is when its first sample arrived.
		uint64_t q = (uint64_t)now.QuadPart;
		if(done && freq)
			q += (uint64_t)((double)(done / frame_bytes) * (double)freq / (double)g_arate);
		uint64_t seq = *A64(24);
		uint8_t *slot = g_abase + FRAMETAP_AUDIO_HEADER + (size_t)(seq % FRAMETAP_AUDIO_SLOT_COUNT) * FRAMETAP_AUDIO_SLOT_SIZE;
		*(volatile uint64_t *)(slot + 0) = UINT64_MAX;
		MemoryBarrier();
		*(volatile uint64_t *)(slot + 8) = q;
		*(volatile uint32_t *)(slot + 16) = (uint32_t)n;
		*(volatile uint32_t *)(slot + 20) = g_arate;
		*(volatile uint32_t *)(slot + 24) = g_achannels;
		memcpy(slot + FRAMETAP_AUDIO_SLOT_HEADER, buf + done, n);
		MemoryBarrier();
		*(volatile uint64_t *)(slot + 0) = seq;
		InterlockedExchange64((volatile LONG64 *)A64(24), (LONG64)(seq + 1));
		done += n;
	}
}

void frametap_end(void)
{
	if(!g_base)
		return;
	// Keep the mapping open (the reader may still hold it); just mark the stream as stopped.
	InterlockedExchange((volatile LONG *)H32(0), 0);
	if(g_abase)
		InterlockedExchange((volatile LONG *)A32(0), 0);
	if(g_event)
		SetEvent(g_event);
}

#elif defined(__APPLE__)
// macOS: the same layout in POSIX shared memory. The Windows name maps to a POSIX one by what follows its
// last backslash ("Local\chiaki-frametap" is "/chiaki-frametap"), as framegen/chiaki_tap.py's posix_name does; macOS
// allows 31 characters with the "/". There's no event: the reader polls the header while it waits. Times are
// nanoseconds of the clock Python's time.monotonic_ns() reads (on macOS mach_absolute_time, which is
// CLOCK_UPTIME_RAW: macOS's CLOCK_MONOTONIC also counts time asleep), so the header's qpc_freq is 1e9. The barriers
// are where the Windows code has MemoryBarrier and Interlocked calls, which are full barriers.
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define TOTAL_SIZE ((size_t)FRAMETAP_HEADER_SIZE + (size_t)FRAMETAP_SLOT_COUNT * FRAMETAP_SLOT_SIZE)
#define AUDIO_TOTAL_SIZE ((size_t)FRAMETAP_AUDIO_HEADER + (size_t)FRAMETAP_AUDIO_SLOT_COUNT * FRAMETAP_AUDIO_SLOT_SIZE)
#define SHM_NAME_MAX 31 // PSHMNAMLEN, with the "/"
#define QPC_FREQ 1000000000ull

static uint8_t *g_base = NULL;
static uint8_t *g_abase = NULL; // the sound ring; NULL if it couldn't be made (the frames still flow)
static uint32_t g_arate = 48000, g_achannels = 2;
static int g_enabled = -1;

#define H32(off) ((volatile uint32_t *)(g_base + (off)))
#define H64(off) ((volatile uint64_t *)(g_base + (off)))
#define A32(off) ((volatile uint32_t *)(g_abase + (off)))
#define A64(off) ((volatile uint64_t *)(g_abase + (off)))
#define BARRIER() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define XCHG(p, v) __atomic_exchange_n((p), (v), __ATOMIC_SEQ_CST)

static uint64_t now_ticks(void)
{
	return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

bool frametap_enabled(void)
{
	if(g_enabled < 0)
	{
		const char *v = getenv("CHIAKI_FRAMETAP");
		g_enabled = (v && v[0] == '1') ? 1 : 0;
	}
	return g_enabled == 1;
}

// "/" + what follows the name's last backslash, leading slashes dropped, + suffix. False if that's empty, has a "/"
// in it or is too long.
static bool shm_name(const char *name, const char *suffix, char *out, size_t cap)
{
	const char *base = strrchr(name, '\\');
	base = base ? base + 1 : name;
	while(*base == '/')
		base++;
	if(!*base || strchr(base, '/'))
		return false;
	int n = snprintf(out, cap, "/%s%s", base, suffix);
	return n > 0 && (size_t)n < cap && n <= SHM_NAME_MAX;
}

// `size` bytes of the shared memory `path`, created if it isn't there. Either side may come first: macOS sets a
// shared-memory object's size only once, while it's still empty, so the second side to try gets EINVAL and finds it
// already sized. Never unlinked: a reader may still have it open, and the next one opens the same memory.
static uint8_t *map_shared(const char *path, size_t size)
{
	int fd = shm_open(path, O_RDWR | O_CREAT, 0600);
	if(fd < 0)
	{
		fprintf(stderr, "frametap: shm_open %s: %s\n", path, strerror(errno));
		return NULL;
	}
	struct stat st;
	if(fstat(fd, &st) == 0 && st.st_size == 0 && ftruncate(fd, (off_t)size) != 0 && errno != EINVAL)
		fprintf(stderr, "frametap: ftruncate %s: %s\n", path, strerror(errno));
	// When both sides size it at once, the one that gets EINVAL can still see 0 bytes for a moment, until the
	// other's ftruncate shows (about 1 in 600 simultaneous opens): give that up to 200 ms.
	for(int i = 0; i < 1000 && fstat(fd, &st) == 0 && st.st_size == 0; i++)
	{
		struct timespec pause = {0, 200000};
		nanosleep(&pause, NULL);
	}
	void *p = MAP_FAILED;
	if(fstat(fd, &st) != 0 || (uint64_t)st.st_size < size) // (macOS rounds the size up to whole pages)
		fprintf(stderr, "frametap: %s is %lld bytes, too small for the frame tap (%zu)\n", path,
			(long long)st.st_size, size);
	else
		p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	return p == MAP_FAILED ? NULL : (uint8_t *)p;
}

static bool open_shared(void)
{
	if(g_base)
		return true;
	const char *name = getenv("CHIAKI_FRAMETAP_NAME");
	if(!name || !name[0] || strlen(name) > 200)
		name = FRAMETAP_NAME;
	char path[256], audio_path[256];
	if(!shm_name(name, "", path, sizeof(path)) || !shm_name(name, FRAMETAP_AUDIO_SUFFIX, audio_path, sizeof(audio_path)))
	{
		fprintf(stderr, "frametap: the name %s doesn't fit this system's shared memory: at most %d characters after "
			"any Local\\, and no \"/\" (\"/\" + the name + \"%s\" has to fit %d)\n", name,
			SHM_NAME_MAX - 1 - (int)strlen(FRAMETAP_AUDIO_SUFFIX), FRAMETAP_AUDIO_SUFFIX, SHM_NAME_MAX);
		return false;
	}
	g_base = map_shared(path, TOTAL_SIZE);
	if(!g_base)
		return false;
	g_abase = map_shared(audio_path, AUDIO_TOTAL_SIZE);
	return true;
}

static void audio_header(uint64_t qpc_freq)
{
	if(!g_abase)
		return;
	XCHG(A32(0), 0u);
	*A32(4) = 1;
	*A32(8) = FRAMETAP_AUDIO_SLOT_COUNT;
	*A32(12) = FRAMETAP_AUDIO_SLOT_SIZE;
	*A32(16) = g_arate;
	*A32(20) = g_achannels;
	*A64(32) = qpc_freq;
	XCHG(A64(24), (uint64_t)0);
	XCHG(A32(0), FRAMETAP_AUDIO_MAGIC);
}

bool frametap_begin(uint32_t codec, uint32_t width, uint32_t height, uint32_t max_fps)
{
	if(!frametap_enabled() || !open_shared())
		return false;
	XCHG(H32(0), 0u); // header is being rewritten
	*H32(4) = FRAMETAP_VERSION;
	*H32(8) = FRAMETAP_SLOT_COUNT;
	*H32(12) = FRAMETAP_SLOT_SIZE;
	*H32(16) = codec;
	*H32(20) = width;
	*H32(24) = height;
	*H32(28) = max_fps;
	*H64(48) = QPC_FREQ;
	*H32(40) = 0;
	*H32(64) = FRAMETAP_FEATURE_PRESS | FRAMETAP_FEATURE_AUDIO_DELAY | (g_abase ? FRAMETAP_FEATURE_AUDIO_TAP : 0u);
	*H32(68) = 0;
	audio_header(QPC_FREQ);
	// The reader treats a new session as a new stream: it resets its decoder and waits for a keyframe.
	XCHG(H64(32), (uint64_t)0);
	__atomic_add_fetch(H32(44), 1u, __ATOMIC_SEQ_CST);
	XCHG(H32(0), FRAMETAP_MAGIC); // last: header is complete
	return true;
}

void frametap_push(const uint8_t *buf, size_t size, int32_t frames_lost, bool recovered)
{
	if(!g_base || *H32(0) != FRAMETAP_MAGIC)
		return;
	uint64_t now = now_ticks();
	uint64_t seq = *H64(32);
	uint8_t *slot = g_base + FRAMETAP_HEADER_SIZE + (size_t)(seq % FRAMETAP_SLOT_COUNT) * FRAMETAP_SLOT_SIZE;
	size_t cap = FRAMETAP_SLOT_SIZE - FRAMETAP_SLOT_HEADER;
	uint32_t flags = 0;
	if(size > cap)
	{
		size = cap;
		flags |= 1;
	}
	// Mark the slot as being rewritten first, so a reader copying it sees the sequence change and retries.
	*(volatile uint64_t *)(slot + 0) = UINT64_MAX;
	BARRIER();
	*(volatile uint64_t *)(slot + 8) = now;
	*(volatile uint32_t *)(slot + 16) = (uint32_t)size;
	*(volatile int32_t *)(slot + 20) = frames_lost;
	*(volatile uint32_t *)(slot + 24) = recovered ? 1u : 0u;
	*(volatile uint32_t *)(slot + 28) = flags;
	memcpy(slot + FRAMETAP_SLOT_HEADER, buf, size);
	BARRIER();
	*(volatile uint64_t *)(slot + 0) = seq;
	XCHG(H64(32), seq + 1);
}

bool frametap_take_idr_request(void)
{
	if(!g_base)
		return false;
	uint32_t expected = 1;
	return __atomic_compare_exchange_n(H32(40), &expected, 0u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

uint32_t frametap_take_press_request(uint32_t *hold_ms)
{
	if(!g_base)
		return 0;
	uint32_t buttons = XCHG(H32(68), 0u);
	if(buttons && hold_ms)
	{
		uint32_t h = *H32(72);
		*hold_ms = h < 20 ? 20 : h > 2000 ? 2000 : h;
	}
	return buttons;
}

void frametap_press_sent(void)
{
	if(!g_base)
		return;
	XCHG(H64(80), now_ticks());
	__atomic_add_fetch(H32(76), 1u, __ATOMIC_SEQ_CST);
}

uint32_t frametap_audio_delay_ms(void)
{
	if(!g_base || *H32(0) != FRAMETAP_MAGIC)
		return 0;
	uint32_t ms = *H32(88);
	uint64_t freq = *H64(48);
	if(!ms || !freq)
		return 0;
	if((int64_t)(now_ticks() - *H64(56)) > (int64_t)(2 * freq)) // SmoovPlay stopped
		return 0;
	return ms > 500 ? 500 : ms;
}

void frametap_audio_format(uint32_t rate, uint32_t channels)
{
	if(!rate || !channels || channels > 8)
		return;
	g_arate = rate;
	g_achannels = channels;
	if(g_abase && *A32(0) == FRAMETAP_AUDIO_MAGIC)
	{
		*A32(16) = rate;
		*A32(20) = channels;
	}
}

void frametap_push_audio(const uint8_t *buf, size_t size)
{
	if(!g_abase || *A32(0) != FRAMETAP_AUDIO_MAGIC || !buf || !size)
		return;
	uint64_t now = now_ticks();
	uint64_t freq = *A64(32);
	size_t frame_bytes = 2u * g_achannels;
	size_t cap = (FRAMETAP_AUDIO_SLOT_SIZE - FRAMETAP_AUDIO_SLOT_HEADER) / frame_bytes * frame_bytes;
	size_t done = 0;
	while(done < size)
	{
		size_t n = size - done < cap ? size - done : cap;
		// A chunk too big for one slot goes in several; each slot's time is when its first sample arrived.
		uint64_t q = now;
		if(done && freq)
			q += (uint64_t)((double)(done / frame_bytes) * (double)freq / (double)g_arate);
		uint64_t seq = *A64(24);
		uint8_t *slot = g_abase + FRAMETAP_AUDIO_HEADER + (size_t)(seq % FRAMETAP_AUDIO_SLOT_COUNT) * FRAMETAP_AUDIO_SLOT_SIZE;
		*(volatile uint64_t *)(slot + 0) = UINT64_MAX;
		BARRIER();
		*(volatile uint64_t *)(slot + 8) = q;
		*(volatile uint32_t *)(slot + 16) = (uint32_t)n;
		*(volatile uint32_t *)(slot + 20) = g_arate;
		*(volatile uint32_t *)(slot + 24) = g_achannels;
		memcpy(slot + FRAMETAP_AUDIO_SLOT_HEADER, buf + done, n);
		BARRIER();
		*(volatile uint64_t *)(slot + 0) = seq;
		XCHG(A64(24), seq + 1);
		done += n;
	}
}

void frametap_end(void)
{
	if(!g_base)
		return;
	// Keep the memory mapped (the reader may still hold it); just mark the stream as stopped.
	XCHG(H32(0), 0u);
	if(g_abase)
		XCHG(A32(0), 0u);
}

#else // neither Windows nor macOS: compiled in, does nothing

bool frametap_enabled(void) { return false; }
bool frametap_begin(uint32_t codec, uint32_t width, uint32_t height, uint32_t max_fps)
{
	(void)codec; (void)width; (void)height; (void)max_fps;
	return false;
}
void frametap_push(const uint8_t *buf, size_t size, int32_t frames_lost, bool recovered)
{
	(void)buf; (void)size; (void)frames_lost; (void)recovered;
}
bool frametap_take_idr_request(void) { return false; }
uint32_t frametap_take_press_request(uint32_t *hold_ms) { (void)hold_ms; return 0; }
uint32_t frametap_audio_delay_ms(void) { return 0; }
void frametap_audio_format(uint32_t rate, uint32_t channels) { (void)rate; (void)channels; }
void frametap_push_audio(const uint8_t *buf, size_t size) { (void)buf; (void)size; }
void frametap_press_sent(void) {}
void frametap_end(void) {}

#endif
