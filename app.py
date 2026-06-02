import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import time


class ToolTrackingSystem:
    def __init__(self):
        self.model = YOLO("best.pt")

        self.expected_tools = {}
        self.tools_in_use = set()
        self.tools_available = set()
        self.all_tools_observed = set()

        self.cap = None

        # 🔥 NEW: stability tracking
        self.last_seen = {}  # tool → last seen timestamp
        self.missing_delay = 2.0  # seconds before marking "in use"
        self.new_tool_buffer = {}  # tool → count before confirming

    def get_tool_names_from_detection(self, results):
        tools_detected = set()

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    conf = float(box.conf[0])

                    # 🔥 Confidence filter
                    if conf < 0.6:
                        continue

                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    tools_detected.add(label)

        return tools_detected

    def update_toolset(self, current_tools):
        new_tools = set()

        for tool in current_tools:
            # 🔥 Require multiple detections before adding new tool
            self.new_tool_buffer[tool] = self.new_tool_buffer.get(tool, 0) + 1

            if self.new_tool_buffer[tool] >= 5:  # seen 5 times
                if tool not in self.all_tools_observed:
                    self.expected_tools[tool] = 1
                    self.all_tools_observed.add(tool)
                    self.tools_available.add(tool)
                    new_tools.add(tool)

        return new_tools

    def process_tool_changes(self, current_tools):
        current_time = time.time()

        # 🔥 Update last seen time
        for tool in current_tools:
            self.last_seen[tool] = current_time

        taken_tools = set()
        returned_tools = set()

        # 🔥 Check missing tools with delay
        for tool in list(self.tools_available):
            if tool not in current_tools:
                last_time = self.last_seen.get(tool, 0)

                if current_time - last_time > self.missing_delay:
                    taken_tools.add(tool)

        # 🔥 Check returned tools
        for tool in current_tools:
            if tool in self.tools_in_use:
                returned_tools.add(tool)

        # 🔥 Update states
        for tool in taken_tools:
            self.tools_in_use.add(tool)
            if tool in self.tools_available:
                self.tools_available.remove(tool)

        for tool in returned_tools:
            self.tools_in_use.remove(tool)
            self.tools_available.add(tool)

        # Add newly seen tools to available
        for tool in current_tools:
            if tool not in self.tools_in_use:
                self.tools_available.add(tool)

        return taken_tools, returned_tools

    def check_missing_tools(self):
        return [
            tool for tool in self.expected_tools
            if tool not in self.tools_available and tool not in self.tools_in_use
        ]

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)

        current_tools = self.get_tool_names_from_detection(results)
        new_tools = self.update_toolset(current_tools)
        taken_tools, returned_tools = self.process_tool_changes(current_tools)

        annotated_frame = results[0].plot()

        return annotated_frame, current_tools, new_tools, taken_tools, returned_tools


def main():
    st.set_page_config(page_title="AI Tool Tracking System", layout="wide")
    st.title("🔧 AI Tool Tracking System")

    if "tracker" not in st.session_state:
        st.session_state.tracker = ToolTrackingSystem()
    if "running" not in st.session_state:
        st.session_state.running = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    if "recent_returned" not in st.session_state:
        st.session_state.recent_returned = set()

    col1, col2, col3 = st.columns(3)
    start = col1.button("🎥 Start Camera")
    stop = col2.button("🛑 Stop Camera")
    reset = col3.button("🔄 Reset")

    if start:
        cap = cv2.VideoCapture(1)  # USB camera
        if not cap.isOpened():
            st.error("Camera not working")
        else:
            st.session_state.tracker.cap = cap
            st.session_state.running = True

    if stop:
        st.session_state.stop_requested = True

    if reset:
        st.session_state.tracker = ToolTrackingSystem()
        st.session_state.running = False
        st.session_state.stop_requested = False
        st.session_state.recent_returned = set()
        st.success("Reset done")

    col_video, col_status = st.columns([2, 1])

    with col_video:
        video_placeholder = st.empty()

    with col_status:
        status_placeholder = st.empty()
        alert_placeholder = st.empty()
        table_placeholder = st.empty()

    if st.session_state.running:
        cap = st.session_state.tracker.cap

        while st.session_state.running:
            ret, frame = cap.read()

            if not ret:
                st.error("Camera error")
                break

            if st.session_state.stop_requested:
                missing = st.session_state.tracker.check_missing_tools()
                in_use = st.session_state.tracker.tools_in_use

                if missing or in_use:
                    alert_placeholder.error(
                        f"⚠️ Cannot stop!\nMissing: {', '.join(missing)}\nIn Use: {', '.join(in_use)}"
                    )
                    st.session_state.stop_requested = False
                else:
                    st.session_state.running = False
                    st.session_state.stop_requested = False
                    cap.release()
                    st.success("Camera stopped")
                    break

            annotated, current, new, taken, returned = \
                st.session_state.tracker.process_frame(frame)

            st.session_state.recent_returned = set(returned)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            missing = st.session_state.tracker.check_missing_tools()

            if missing:
                alert_placeholder.error(f"⚠️ Missing: {', '.join(missing)}")
            else:
                alert_placeholder.success("✅ All tools OK")

            status_placeholder.markdown(f"""
            **Total Tools:** {len(st.session_state.tracker.expected_tools)}  
            **Available:** {len(st.session_state.tracker.tools_available)}  
            **In Use:** {len(st.session_state.tracker.tools_in_use)}
            """)

            data = []
            for tool in st.session_state.tracker.expected_tools:
                if tool in st.session_state.recent_returned:
                    status = "🔁 Returned"
                elif tool in st.session_state.tracker.tools_available:
                    status = "✅ Available"
                elif tool in st.session_state.tracker.tools_in_use:
                    status = "🔧 In Use"
                else:
                    status = "❌ Missing"

                data.append({"Tool": tool, "Status": status})

            if data:
                table_placeholder.dataframe(pd.DataFrame(data), use_container_width=True)

            time.sleep(0.03)

    else:
        st.info("Click 'Start Camera' to begin")


if __name__ == "__main__":
    main()