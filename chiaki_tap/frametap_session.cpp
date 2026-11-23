// See frametap_session.h. The frame tap's work inside chiaki-ng's StreamSession: copying frames and sound out,
// SmoovPlay's keyframe requests, the delay test's button presses and the sound delay. It reaches the session's
// private parts as a friend (one line in streamsession.h).
#include <frametap_session.h>

#include <frametap.h>
#include <streamsession.h>

#include <QMetaObject>
#include <QTimer>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cstdlib>
#include <cstring>
#endif

bool FrameTapSession::Begin(StreamSession *session, ChiakiSession *chiaki_session, const ChiakiConnectVideoProfile &profile)
{
	if(!frametap_begin((uint32_t)profile.codec, profile.width, profile.height, profile.max_fps))
		return false;
#ifdef _WIN32
	CHIAKI_LOGI(session->GetChiakiLog(), "Frame tap on: copying every video frame to shared memory " FRAMETAP_NAME);
#else
	// POSIX shared memory is "/" and what follows the name's last backslash (frametap_posix.h)
	const char *name = getenv("CHIAKI_FRAMETAP_NAME");
	if(!name || !name[0] || strlen(name) > 200)
		name = FRAMETAP_NAME;
	const char *leaf = strrchr(name, '\\');
	leaf = leaf ? leaf + 1 : name;
	while(*leaf == '/')
		leaf++;
	CHIAKI_LOGI(session->GetChiakiLog(), "Frame tap on: copying every video frame to shared memory /%s", leaf);
#endif
	chiaki_session_set_video_sample_cb(chiaki_session, VideoSample, session);
#ifdef _WIN32
	// SmoovPlay's video window opens as the stream starts, while chiaki-ng (where the player just pressed connect)
	// is in front, and Windows only lets a window come to the front if the program in front allows it.
	AllowSetForegroundWindow(ASFW_ANY);
#endif
	return true;
}

void FrameTapSession::End()
{
	frametap_end();
}

bool FrameTapSession::VideoSample(uint8_t *buf, size_t buf_size, int32_t frames_lost, bool frame_recovered, void *user)
{
	auto session = reinterpret_cast<StreamSession *>(user);
	// Copy first (never blocks), then decode as usual. A reader that joined mid-stream or fell behind asks for a
	// keyframe; RequestIDR runs on the GUI thread, like chiaki-ng's own calls to it.
	frametap_push(buf, buf_size, frames_lost, frame_recovered);
	if(frametap_take_idr_request())
		QMetaObject::invokeMethod(session, [session]() { session->RequestIDR(); }, Qt::QueuedConnection);
	uint32_t hold_ms = 0;
	if(uint32_t buttons = frametap_take_press_request(&hold_ms))
		QMetaObject::invokeMethod(session, [session, buttons, hold_ms]() { Press(session, buttons, hold_ms); }, Qt::QueuedConnection);
	return chiaki_ffmpeg_decoder_video_sample_cb(buf, buf_size, frames_lost, frame_recovered, session->ffmpeg_decoder);
}

// SmoovPlay's delay test: press the buttons, send the state at once, note the time, let go after hold_ms.
void FrameTapSession::Press(StreamSession *session, uint32_t buttons, uint32_t hold_ms)
{
	session->keyboard_state.buttons |= buttons;
	session->SendFeedbackState();
	frametap_press_sent();
	QTimer::singleShot((int)hold_ms, session, [session, buttons]() {
		session->keyboard_state.buttons &= ~buttons;
		session->SendFeedbackState();
	});
}

// Sound held back by SmoovPlay's picture delay (frametap_audio_delay_ms): a first-in first-out buffer of exactly
// that much sound. Each frame in sends one frame's worth out, so the device is fed as steadily as before; while the
// delay grows that's silence, and when it shrinks the oldest sound is skipped.
static QByteArray &TapAudioFifo() { static QByteArray fifo; return fifo; }
static size_t &TapAudioBytesPerMs() { static size_t n = 192; return n; }  // 48 kHz stereo 16-bit
static size_t &TapAudioSampleSize() { static size_t n = 4; return n; }

void FrameTapSession::AudioFormat(unsigned int rate, size_t sample_size)
{
	TapAudioFifo().clear();
	TapAudioSampleSize() = sample_size ? sample_size : 4;
	TapAudioBytesPerMs() = (size_t)rate * TapAudioSampleSize() / 1000;
	frametap_audio_format(rate, (uint32_t)(TapAudioSampleSize() / sizeof(int16_t)));
}

void FrameTapSession::QueueAudio(StreamSession *session, const QByteArray &buf)
{
	QByteArray &fifo = TapAudioFifo();
	size_t ss = TapAudioSampleSize();
	size_t want = (size_t)frametap_audio_delay_ms() * TapAudioBytesPerMs() / ss * ss;
	if(!want && fifo.isEmpty())
	{
		session->QueueAudioOutData(buf);
		return;
	}
	fifo.append(buf);
	size_t n = (size_t)buf.size();
	if((size_t)fifo.size() > want + 2 * n)  // the delay shrank: skip the oldest sound
		fifo.remove(0, (int)((size_t)fifo.size() - want - n));
	if((size_t)fifo.size() >= want + n)
	{
		session->QueueAudioOutData(fifo.left((int)n));
		fifo.remove(0, (int)n);
	}
	else  // the delay is growing: silence while the buffer fills
		session->QueueAudioOutData(QByteArray((int)n, 0));
}

void FrameTapSession::PushAudio(StreamSession *session, const int16_t *buf, size_t samples_count)
{
	if(session->audio_out_sample_size)
		frametap_push_audio((const uint8_t *)buf, samples_count * session->audio_out_sample_size);
}
