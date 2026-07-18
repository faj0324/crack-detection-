#!/usr/bin/env python3
"""Run YOLO11 crack detection and publish results to MQTT for Node-RED.

Reads from a webcam (default) or --source (video file / image folder),
runs inference with models/best.pt, shows a preview window with boxes,
and publishes each frame's detections as JSON to an MQTT topic that
Node-RED subscribes to (mirrors the IVIS Label Studio + MQTT setup).

Payload on 'ivis/crack/detections':
    {"timestamp": <epoch>, "count": N,
     "detections": [{"class": "crack", "confidence": 0.9,
                     "bbox": [x1, y1, x2, y2]}]}

Uses paho-mqtt v2 API. Publishes a status message on connect and a
heartbeat every ~10s on 'ivis/crack/status' so Node-RED can show the
detector as online/offline. Detections throttled to ~2 publishes/sec.

Example:
    python detect_mqtt.py --weights ../models/best.pt --broker localhost
"""
import argparse
import json
import time

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO

STATUS_TOPIC = "ivis/crack/status"
STATUS_INTERVAL = 10.0  # seconds between heartbeat status messages


def publish_status(client: mqtt.Client, status: str) -> None:
    """Publish a retained status/heartbeat JSON on the status topic."""
    client.publish(
        STATUS_TOPIC,
        json.dumps({"status": status, "timestamp": time.time()}),
        retain=True,
    )


def on_connect(client, userdata, flags, reason_code, properties):
    """paho v2 callback: announce ourselves once connected."""
    print(f"MQTT connected (rc={reason_code})")
    publish_status(client, "online")


def make_client(broker: str, port: int) -> mqtt.Client:
    """Create and connect a paho-mqtt v2 client with a background loop."""
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    # Last-will so Node-RED sees us drop if the process dies.
    client.will_set(STATUS_TOPIC, json.dumps({"status": "offline"}), retain=True)
    client.connect(broker, port, keepalive=60)
    client.loop_start()
    return client


def build_payload(result) -> dict:
    """Convert one Ultralytics result into the JSON payload dict."""
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
        detections.append({
            "class": result.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [x1, y1, x2, y2],
        })
    return {
        "timestamp": time.time(),
        "count": len(detections),
        "detections": detections,
    }


def main():
    ap = argparse.ArgumentParser(description="YOLO11 crack detect -> MQTT.")
    ap.add_argument("--weights", default="models/best.pt", help="Trained weights")
    ap.add_argument("--source", default="0", help="Webcam index, video file, or image folder")
    ap.add_argument("--broker", default="localhost", help="MQTT broker host")
    ap.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    ap.add_argument("--topic", default="ivis/crack/detections", help="Detections topic")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    ap.add_argument("--rate", type=float, default=2.0, help="Max publishes per second")
    ap.add_argument("--no-preview", action="store_true", help="Disable the preview window")
    args = ap.parse_args()

    model = YOLO(args.weights)
    client = make_client(args.broker, args.port)

    # A digit source means a webcam index; otherwise a path (Ultralytics handles both).
    source = int(args.source) if args.source.isdigit() else args.source
    min_interval = 1.0 / args.rate if args.rate > 0 else 0.0
    last_pub = 0.0
    last_status = time.time()  # on_connect already sent the first "online"

    print(f"Streaming {source!r} -> topic '{args.topic}' (Ctrl-C / 'q' to stop)")
    try:
        # stream=True yields results frame-by-frame without buffering everything.
        for result in model.predict(source=source, conf=args.conf, stream=True, verbose=False):
            if not args.no_preview:
                cv2.imshow("crack-detection", result.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.time()
            if now - last_pub >= min_interval:
                payload = build_payload(result)
                client.publish(args.topic, json.dumps(payload))
                last_pub = now
                print(f"[{time.strftime('%H:%M:%S')}] count={payload['count']}")

            # Heartbeat so Node-RED can flip to "offline" if these stop arriving.
            if now - last_status >= STATUS_INTERVAL:
                publish_status(client, "online")
                last_status = now
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        publish_status(client, "offline")
        client.loop_stop()
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
