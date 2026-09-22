"""Send a TypeSafe-compatible System One request to the Blocks agent."""

import base64
import json
import threading

from blocks_network import SendMessageRequestPart, create_task_client


def main():
    client = create_task_client()

    session = client.send_message(
        agent_name="jev_open_source",
        request_parts=[SendMessageRequestPart(part_id="request", text=json.dumps({
            "state": "I was charged twice for order A-104. Please refund the duplicate.",
            "questions": {
                "refund": {
                    "type": "noul",
                    "instructions": "Does the text request a refund?",
                }
            },
        }))],
    )

    print(f"Task created: {session.task_id}")

    done = threading.Event()

    def on_progress(event):
        print("[progress]", event.get("message") or event.get("progress") or "")

    def on_artifact(event):
        ref = event.artifact_ref
        if ref is None:
            print("[artifact]", event.raw)
            return
        if ref.kind == "inline" and ref.data:
            text = base64.b64decode(ref.data).decode()
            print("[artifact]", text)
        else:
            downloaded = session.download_artifact(ref)
            print("[artifact]", downloaded.data.decode())

    def on_terminal(event):
        print("[done] Task complete")
        done.set()

    session.on_progress(on_progress)
    session.on_artifact(on_artifact)
    session.on_terminal(on_terminal)

    done.wait(timeout=60)
    session.close()
    client.destroy()


if __name__ == "__main__":
    main()
