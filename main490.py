import cv2
import time
import math as m
import mediapipe as mp


# math helpers
# these are small, reusable functions so the main loop stays readable

def findDistance(x1, y1, x2, y2):
    # euclidean distance between two points
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def clamp(value, lo, hi):
    # prevents floating point drift from breaking acos by forcing the value into a safe range
    return max(lo, min(hi, value))


def inclination_deg(x1, y1, x2, y2):
    """
    computes how tilted a body segment is relative to vertical "up"

    segment is from (x1,y1) -> (x2,y2)
    0°  = perfectly vertical (upright)
    bigger number = more leaning/tilting away from vertical

    why this works:
    - treat the segment as a vector v = (dx, dy)
    - compare it to vertical up vector u = (0, -1) in image coords
    - cos(theta) = (v dot u) / (|v||u|)
      since |u| = 1 and v dot u = (dx*0 + dy*-1) = -dy
      cos(theta) = (-dy) / |v|
    - theta = acos(cos(theta)) converted to degrees
    """
    dx = x2 - x1
    dy = y2 - y1

    mag = m.sqrt(dx * dx + dy * dy)
    if mag < 1e-6:
        # guard: if two points are basically the same, we can't define a direction
        return 0.0

    # vertical up in image coordinates is (0, -1)
    cos_theta = (-dy) / mag
    cos_theta = clamp(cos_theta, -1.0, 1.0)

    return (180.0 / m.pi) * m.acos(cos_theta)


def sendWarning():
    # placeholder for later (sound, notification, etc.)
    pass


# settings / tuning knobs
# color palette (bgr). these only affect visuals, not logic.

GOOD = (120, 200, 180)       # teal (good posture)
BAD = (160, 120, 90)         # muted indigo (bad posture)
ACCENT = (210, 210, 210)     # soft gray (labels/info)
LANDMARK = (160, 210, 240)   # warm sand (points)
SECONDARY = (220, 170, 220)  # lavender (right shoulder / extra point)
TEXT_DARK = (60, 60, 60)     # dark gray (optional if you want)

# text readability fix
# these help the ui stay readable on bright backgrounds
TEXT_PRIMARY = (245, 245, 245)    # near white
TEXT_SECONDARY = (180, 180, 180)  # soft gray
TEXT_SHADOW = (0, 0, 0)           # black shadow

font = cv2.FONT_HERSHEY_SIMPLEX


def draw_text(image, text, org, scale=0.9, color=TEXT_PRIMARY, thickness=2):
    # draws outlined/shadowed text so it stays readable on any background
    # shadow first
    cv2.putText(image, text, (org[0] + 2, org[1] + 2), font, scale, TEXT_SHADOW, thickness + 2)
    # main text
    cv2.putText(image, text, org, font, scale, color, thickness)


# mediapipe pose detection object
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# step 3: hysteresis + hold-time (anti-flicker) ----------------
# goal: prevent rapid "good/bad" toggling due to tiny measurement jitter.
# how: we require a condition to hold for a short time before switching states.

posture_state = "bad"        # current posture state ("good" or "bad")
switch_hold_s = 0.0          # how long the opposite condition has been true
SWITCH_HOLD_SECONDS = 0.50   # must hold this long to switch states

# posture thresholds (smaller inclination is better)
# enter_good is stricter (harder) than stay_good (looser).
# this creates a buffer zone that reduces flicker.

NECK_ENTER_GOOD = 35.0
TORSO_ENTER_GOOD = 8.0

NECK_EXIT_GOOD = 42.0
TORSO_EXIT_GOOD = 12.0

# step 4: dt-based timekeeping
# for prerecorded video, dt should match the video timeline (not wall-clock).
# we use dt = 1/fps so "good time" and "adjust time" are objectively correct for the mp4.

good_seconds = 0.0
bad_seconds = 0.0

# landmark confidence gate
# mediapipe gives a "visibility" value per landmark.
# if confidence is low, angles can jump randomly, so we skip those frames.

VIS_MIN = 0.50

# step 5: ema smoothing
# ema reduces jitter by blending new measurements with the previous smoothed value.
# alpha controls responsiveness:
# - lower alpha (0.10) = smoother but slower to react
# - higher alpha (0.30) = reacts faster but more jittery

EMA_ALPHA = 0.20
neck_ema = None
torso_ema = None


if __name__ == "__main__":
    # input source: mp4 file.
    file_name = 'input23.mp4'
    cap = cv2.VideoCapture(file_name)

    # video output setup (keep same size/fps as input when possible)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # force a safe fps if metadata is missing or looks wrong (common on hevc/iphone mp4)
    if fps is None or fps <= 1 or fps > 120:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # if fps is missing/0, fall back to 30
    fps_out = fps if fps and fps > 0 else 30.0
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps_out, frame_size)

    # fixed dt for prerecorded video (seconds per frame)
    dt_per_frame = 1.0 / fps_out

    while cap.isOpened():
        # frame acquisition
        success, image = cap.read()
        if not success:
            # end of file
            break

        # dt for this frame (video-time, not processing-time)
        dt = dt_per_frame

        h, w = image.shape[:2]

        # pose inference -
        # important: mediapipe expects rgb input
        # also important: don't overwrite the original bgr 'image' we draw on
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image_rgb)

        if not keypoints.pose_landmarks:
            # no person detected -> write/display original frame and continue
            video_output.write(image)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue

        # landmark access helpers
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # landmark coordinates -
        # we use: left shoulder, right shoulder, left ear, left hip
        # note: these are pixel coordinates (scaled by width/height)

        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # landmark visibilities
        # visibility is a confidence-ish score from mediapipe (0 to 1)
        ear_vis = lm.landmark[lmPose.LEFT_EAR].visibility
        sh_vis = lm.landmark[lmPose.LEFT_SHOULDER].visibility
        hip_vis = lm.landmark[lmPose.LEFT_HIP].visibility

        # alignment indicator (side view hint)
        # if shoulders are far apart, you're likely facing camera; if close, you're more side-on
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        if offset < 100:
            draw_text(image, f"{int(offset)} aligned", (w - 180, 30), 0.9, TEXT_PRIMARY, 2)
        else:
            draw_text(image, f"{int(offset)} not aligned", (w - 220, 30), 0.9, TEXT_PRIMARY, 2)

        # confidence gating
        # skip frames where landmarks are unreliable to avoid random angle spikes
        if ear_vis < VIS_MIN or sh_vis < VIS_MIN or hip_vis < VIS_MIN:
            draw_text(image, "low confidence (skipping)", (10, 60), 0.9, TEXT_SECONDARY, 2)
            video_output.write(image)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue

        # posture angles
        # raw angles from current frame (no smoothing yet)

        # neck metric: shoulder -> ear tilt vs vertical
        neck_raw = inclination_deg(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)

        # torso metric: hip -> shoulder tilt vs vertical
        torso_raw = inclination_deg(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # step 5: ema smoothing
        # initialize on first valid frame, then smooth afterwards
        if neck_ema is None:
            neck_ema = neck_raw
            torso_ema = torso_raw
        else:
            neck_ema = EMA_ALPHA * neck_raw + (1.0 - EMA_ALPHA) * neck_ema
            torso_ema = EMA_ALPHA * torso_raw + (1.0 - EMA_ALPHA) * torso_ema

        # use smoothed values for classification (less jitter)
        neck_inclination = neck_ema
        torso_inclination = torso_ema

        # step 3: hysteresis state machine
        # enter_good is strict; stay_good is looser
        enter_good = (neck_inclination <= NECK_ENTER_GOOD) and (torso_inclination <= TORSO_ENTER_GOOD)
        stay_good = (neck_inclination <= NECK_EXIT_GOOD) and (torso_inclination <= TORSO_EXIT_GOOD)

        if posture_state == "good":
            # we're currently good; only switch to bad if we fail stay_good long enough
            if not stay_good:
                switch_hold_s += dt
                if switch_hold_s >= SWITCH_HOLD_SECONDS:
                    posture_state = "bad"
                    switch_hold_s = 0.0
            else:
                switch_hold_s = 0.0
        else:
            # we're currently bad; only switch to good if we satisfy enter_good long enough
            if enter_good:
                switch_hold_s += dt
                if switch_hold_s >= SWITCH_HOLD_SECONDS:
                    posture_state = "good"
                    switch_hold_s = 0.0
            else:
                switch_hold_s = 0.0

        # step 4: timekeeping
        # we accumulate dt while staying in a state.
        if posture_state == "good":
            good_seconds += dt
            bad_seconds = 0.0
            color = GOOD
            status_color = GOOD
            label = "GOOD"
        else:
            bad_seconds += dt
            good_seconds = 0.0
            color = BAD
            status_color = BAD
            label = "ADJUST"

        # ui overlay
        # show raw -> smoothed so you can see what ema is doing
        angle_text_string = (
            f"neck: {neck_raw:.1f} -> {neck_inclination:.1f}    "
            f"torso: {torso_raw:.1f} -> {torso_inclination:.1f}"
        )

        draw_text(image, angle_text_string, (10, 30), 0.8, TEXT_PRIMARY, 2)
        draw_text(image, f"posture: {label}", (10, 60), 0.9, status_color, 2)

        if posture_state == "good":
            draw_text(image, f"good time: {good_seconds:.1f}s", (10, h - 20), 0.9, GOOD, 2)
        else:
            draw_text(image, f"adjust time: {bad_seconds:.1f}s", (10, h - 20), 0.9, BAD, 2)

        # draw landmarks / skeleton
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, LANDMARK, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, LANDMARK, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, LANDMARK, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, SECONDARY, -1)

        # lines: shoulder->ear (neck segment) and hip->shoulder (torso segment)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), color, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), color, 4)

        # warning trigger
        # warning after 180s in "bad" (continuous)
        if bad_seconds > 180:
            sendWarning()

        # output + display
        video_output.write(image)

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # cleanup-
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()
