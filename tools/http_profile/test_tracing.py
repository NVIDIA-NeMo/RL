"""ASGI trace tracks must keep concurrent requests independent."""

import asyncio

from tools.http_profile.tracing import G_REQUEST_TRACK, Trace, TraceHTTPMiddleware


def test_concurrent_requests_keep_separate_tracks_and_restore_context() -> None:
    trace = Trace(True, "test/backend")

    async def exercise() -> None:
        both_entered = asyncio.Event()
        tracks = []

        async def app(scope, receive, send):
            tracks.append(G_REQUEST_TRACK.get())
            if len(tracks) == 2:
                both_entered.set()
            await both_entered.wait()
            await receive()
            await send({"type": "http.response.body", "body": b"ok"})

        async def receive():
            return {"type": "http.request", "body": b"private prompt"}

        async def send(message):
            await asyncio.sleep(0)

        middleware = TraceHTTPMiddleware(app, trace)
        await asyncio.gather(
            *(
                middleware(
                    {
                        "type": "http",
                        "path": "/v1/chat/completions",
                        "headers": [(b"x-profile-call", key)],
                    },
                    receive,
                    send,
                )
                for key in (b"first", b"second")
            )
        )
        assert len(set(tracks)) == 2
        assert G_REQUEST_TRACK.get() == 0

    asyncio.run(exercise())
    spans = [event for event in trace.events if event["name"] == "HTTP request"]
    assert len(spans) == 2
    assert all(event["args"]["received_bytes"] == 14 for event in spans)
    assert all(event["args"]["sent_bytes"] == 2 for event in spans)
    assert "private prompt" not in str(trace.events)
