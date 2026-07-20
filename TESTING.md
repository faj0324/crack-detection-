# Testing it without a webcam

You don't need a camera to demo this. You can just point the detector at a
folder of test images and it works the same way. I usually have four terminals
open: one for the broker, one to watch the messages, one for the detector, and
one for Node-RED. These commands assume you are in the project root with the
venv active and `models/best.pt` in place.

---

## 1. Start the broker

```bash
mosquitto -v
# or if it's installed as a service:
#   brew services start mosquitto        (macOS)
#   sudo systemctl start mosquitto       (Linux)
```

The `-v` just makes it print the traffic so you can see it working. It runs on
`localhost:1883`.

## 2. Watch the messages (optional but handy)

In another terminal, subscribe to the topics so you can see the JSON coming
through before you even open Node-RED:

```bash
mosquitto_sub -h localhost -t 'ivis/crack/#' -v
```

You should see a `ivis/crack/status` message saying `online` as soon as the
detector connects, another one every 10 seconds (the heartbeat), and a stream of
`ivis/crack/detections` messages while it runs. If those show up, everything is
working and Node-RED will work too.

## 3. Run the detector on a folder of images

Point `--source` at a folder of `.jpg` or `.png` images. There are some test
images included in `testimgs/` so this works straight after cloning.
`--no-preview` means it won't try to open a window, which is useful over SSH or
if there is no display:

```bash
python scripts/detect_mqtt.py \
  --weights models/best.pt \
  --source testimgs \
  --broker localhost \
  --conf 0.5 \
  --no-preview
```

One thing to know: the script goes through the folder once and then exits, so
you get a short burst of messages. If you want it to keep going for a demo, just
run it again (or loop it in the shell).

It sends the results to `ivis/crack/detections` about twice a second and keeps
the status on `online`. When you stop it with Ctrl-C it sends `offline`, and even
if it crashes, Node-RED flips to offline by itself after 30 seconds of no
messages.

## 4. Import the flow and open the dashboard

```bash
node-red        # http://localhost:1880
```

1. **Menu → Import**, paste in `nodered/flow.json`, click **Import**, then
   **Deploy**.
2. Open the dashboard at **http://localhost:1880/ui**.

You should see the crack count gauge, the confidence chart, the alert that turns
red when there is a crack, and the detector online/offline box that follows the
heartbeat.

## 5. No detector running? Test just the dashboard

In the Node-RED editor, click the button on the **"test: 2 cracks"** inject node.
It pushes a fake message through the same path so you can check the dashboard
without a broker or a camera. The debug panel on the right shows the message too.
(The online/offline box still needs a real status message from step 3, otherwise
it just reads offline after 30 seconds.)

---

## If something isn't working

| Problem                          | Probably because                                          |
| -------------------------------- | --------------------------------------------------------- |
| Nothing in `mosquitto_sub`       | Broker isn't running, or wrong `--broker`/`--port`.       |
| Dashboard is blank               | The `node-red-dashboard` palette isn't installed.         |
| Detector shows "offline"         | The detector isn't running or stopped more than 30s ago.  |
| `best.pt` not found              | Put the trained weights at `models/best.pt`.              |
