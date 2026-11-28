// The handful of Win32 calls frametap.c uses, on Linux and macOS, so its Windows code builds there unchanged.
//
//   named file mapping       -> POSIX shared memory: "Local\chiaki-frametap" is /dev/shm/chiaki-frametap on Linux
//                               and shm_open("/chiaki-frametap") on macOS
//   QueryPerformanceCounter  -> CLOCK_MONOTONIC in nanoseconds (the clock time.perf_counter() reads on Linux),
//                               CLOCK_UPTIME_RAW on macOS (its time.perf_counter() clock, mach_absolute_time)
//   Interlocked*, barriers   -> GCC/Clang __atomic builtins
//   the named auto-reset event -> a futex on write_seq's low 32 bits (header offset 32): SetEvent wakes the reader,
//                               which sleeps in FUTEX_WAIT there (framegen/tapsys.py). No name, nothing to open.
//                               macOS has no futex, so SetEvent does nothing there and the reader polls write_seq.
//
// macOS's shared memory differs in two more ways, handled in its CreateFileMappingA below. Names are at most 31
// characters with the slash. And an object can be sized only once, while it's still empty.
//
// Included by frametap.c only, in place of <windows.h>. CreateEventA is a macro naming frametap.c's g_base, which is
// in scope where it is used.
#ifndef FRAMETAP_POSIX_H
#define FRAMETAP_POSIX_H

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <errno.h>
#else
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

typedef void *HANDLE;
typedef uint32_t DWORD;
typedef int32_t LONG;
typedef int64_t LONG64;
typedef union { int64_t QuadPart; } LARGE_INTEGER;

#define FALSE 0
#define INVALID_HANDLE_VALUE ((HANDLE)(intptr_t)-1)
#define PAGE_READWRITE 0x04
#define FILE_MAP_ALL_ACCESS 0xF001F

#define InterlockedExchange(p, v) __atomic_exchange_n((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedExchange64(p, v) __atomic_exchange_n((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedIncrement(p) __atomic_add_fetch((p), 1, __ATOMIC_SEQ_CST)
#define MemoryBarrier() __atomic_thread_fence(__ATOMIC_SEQ_CST)

static inline LONG frametap_cas32(volatile LONG *p, LONG exchange, LONG comparand)
{
	__atomic_compare_exchange_n(p, &comparand, exchange, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	return comparand; // the value before, as InterlockedCompareExchange returns
}
#define InterlockedCompareExchange(p, exchange, comparand) frametap_cas32((p), (exchange), (comparand))

static inline int QueryPerformanceFrequency(LARGE_INTEGER *f)
{
	f->QuadPart = 1000000000;
	return 1;
}

static inline int QueryPerformanceCounter(LARGE_INTEGER *v)
{
#ifdef __APPLE__
	// Not CLOCK_MONOTONIC, which on macOS also counts the time the Mac was asleep
	v->QuadPart = (int64_t)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	v->QuadPart = (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#endif
	return 1;
}

// A mapping handle is the shm file descriptor + 1, so NULL still means "failed".
#ifdef __APPLE__
#define FRAMETAP_SHM_NAME_MAX 31 // macOS's PSHMNAMLEN, with the slash

static inline HANDLE CreateFileMappingA(HANDLE file, void *sa, DWORD protect, DWORD size_high, DWORD size_low,
	const char *name)
{
	(void)file; (void)sa; (void)protect;
	// "/" and what follows the last backslash, so "Local\x", "/x" and "x" are all "/x", as in framegen/tapsys.py
	const char *leaf = strrchr(name, '\\');
	leaf = leaf ? leaf + 1 : name;
	while(*leaf == '/')
		leaf++;
	char path[1 + 256]; // "/" + the longest name frametap.c passes (its own 256-byte buffers)
	int n = snprintf(path, sizeof(path), "/%s", leaf);
	if(!*leaf || strchr(leaf, '/') || n > FRAMETAP_SHM_NAME_MAX)
	{
		fprintf(stderr, "frametap: can't use %s as shared memory. On macOS its name is \"/\" and what follows the last "
			"backslash, with no other \"/\" and %d characters at most\n", name, FRAMETAP_SHM_NAME_MAX);
		return NULL;
	}
	int fd = shm_open(path, O_RDWR | O_CREAT, 0600); // either side may create it first
	if(fd < 0)
	{
		fprintf(stderr, "frametap: shm_open %s: %s\n", path, strerror(errno));
		return NULL;
	}
	off_t size = (off_t)(((uint64_t)size_high << 32) | size_low);
	struct stat st = {0};
	// macOS sizes an object only while it's empty, so only the side that comes first sizes it. If both try at once,
	// the second gets EINVAL and can still see 0 bytes for a moment, so it waits up to 200 ms for the size to show.
	if(fstat(fd, &st) == 0 && st.st_size == 0 && ftruncate(fd, size) != 0 && errno != EINVAL)
		fprintf(stderr, "frametap: ftruncate %s: %s\n", path, strerror(errno));
	for(int i = 0; i < 1000 && fstat(fd, &st) == 0 && st.st_size == 0; i++)
	{
		struct timespec pause = {0, 200000};
		nanosleep(&pause, NULL);
	}
	if(fstat(fd, &st) != 0 || st.st_size < size) // macOS rounds the size up to whole pages
	{
		fprintf(stderr, "frametap: %s is %lld bytes, too small for the frame tap (%lld)\n", path, (long long)st.st_size,
			(long long)size);
		close(fd);
		return NULL;
	}
	return (HANDLE)(intptr_t)(fd + 1);
}
#else
static inline HANDLE CreateFileMappingA(HANDLE file, void *sa, DWORD protect, DWORD size_high, DWORD size_low,
	const char *name)
{
	(void)file; (void)sa; (void)protect;
	const char *leaf = strrchr(name, '\\');
	char path[1 + 256]; // "/" + the longest name frametap.c passes (its own 256-byte buffers)
	snprintf(path, sizeof(path), "/%s", leaf ? leaf + 1 : name);
	int fd = shm_open(path, O_RDWR | O_CREAT, 0600); // either side may create it first
	if(fd < 0)
		return NULL;
	off_t size = (off_t)(((uint64_t)size_high << 32) | size_low);
	struct stat st;
	if(fstat(fd, &st) != 0 || (st.st_size < size && ftruncate(fd, size) != 0))
	{
		close(fd);
		return NULL;
	}
	return (HANDLE)(intptr_t)(fd + 1);
}
#endif

static inline void *MapViewOfFile(HANDLE map, DWORD access, DWORD offset_high, DWORD offset_low, size_t size)
{
	(void)access; (void)offset_high; (void)offset_low;
	void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, (int)(intptr_t)map - 1, 0);
	return p == MAP_FAILED ? NULL : p;
}

static inline int CloseHandle(HANDLE h)
{
	return close((int)(intptr_t)h - 1) == 0;
}

// The event is write_seq's low word in the mapping; frametap.c creates it right after mapping g_base.
#define CreateEventA(sa, manual_reset, initial_state, name) ((void)(name), (HANDLE)(g_base + 32))

static inline int SetEvent(HANDLE event)
{
#ifdef __APPLE__
	(void)event; // macOS has no futex, so the reader polls write_seq instead
#else
	// A shared futex (not FUTEX_PRIVATE): the reader is another process. Nothing waiting costs one syscall.
	syscall(SYS_futex, (uint32_t *)event, FUTEX_WAKE, 1, NULL, NULL, 0);
#endif
	return 1;
}

#endif
