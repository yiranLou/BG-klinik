import opensim as osim
from pathlib import Path
import os
import sys
import math


class Conf:
    """Simplified configuration class for testing purposes"""

    def __init__(self, project_path):
        self.project_path = Path(project_path)
        # Simulated configuration data
        self.data = {
            "test_participant": {
                "mass": 70.0,
                "height": 175.0,
                "age": 30
            }
        }

    def get_conf_field(self, participant, fields):
        if participant not in self.data:
            raise ValueError(f"Participant {participant} not found in configuration")

        result = self.data[participant]
        for field in fields:
            if field in result:
                result = result[field]
            else:
                raise ValueError(f"Field {field} not found in participant configuration")

        return result


class Scale:
    """
    Scale tool in pyosim - Scaling tool in pyosim.
    """

    def __init__(
            self,
            model_input,
            model_output,
            xml_input,
            xml_output,
            static_path,
            mass_input,
            height=-1,
            age=-1,
            add_model=None,
            remove_unused=False,
    ):
        self.model_input = model_input
        self.model_output = model_output
        self.model_with_markers_output = model_output.replace(".osim", "_markers.osim")
        self.xml_output = xml_output
        self.static_path = static_path
        self.mass_input = mass_input

        # Load model
        self.model = osim.Model(model_input)

        # Generate ScaleTool instance
        self.scale_tool = osim.ScaleTool(xml_input)
        self.set_anthropometry(mass_input, height, age)
        self.scale_tool.getGenericModelMaker().setModelFileName(model_input)

        # Set time range (reference: static marker data)
        self.time_range = self.time_range_from_static()

        # First add geometry file search paths to avoid .vtp file not found warnings
        orig_geom_dir = os.path.join(os.path.dirname(os.path.abspath(model_input)), "Geometry")
        out_geom_dir = os.path.join(os.path.dirname(os.path.abspath(model_output)), "Geometry")
        osim.ModelVisualizer.addDirToGeometrySearchPaths(orig_geom_dir)
        osim.ModelVisualizer.addDirToGeometrySearchPaths(out_geom_dir)

        # 1. Run model scaling (ModelScaler)
        self.run_model_scaler(mass_input)
        # 2. Manually calculate error between scaled model and static markers
        self.compute_marker_error(
            model_path=self.model_output,
            trc_path=self.static_path,
            label="Model Scaling (ModelScaler) Error"
        )

        # 3. Marker placement (MarkerPlacer)
        self.run_marker_placer()
        # 4. Manually calculate error between model with placed markers and static markers
        self.compute_marker_error(
            model_path=self.model_with_markers_output,
            trc_path=self.static_path,
            label="Marker Placement (MarkerPlacer) Error"
        )

        # 5. Optional: combine other models
        if add_model:
            self.combine_models(add_model)

        # 6. Optional: add back unused markers from original model
        if not remove_unused:
            self.add_unused_markers()

    def time_range_from_static(self):
        """Return [start_time, end_time]"""
        md = osim.MarkerData(self.static_path)
        initial_time = md.getStartFrameTime()
        final_time = md.getLastFrameTime()
        range_time = osim.ArrayDouble()
        range_time.set(0, initial_time)
        range_time.set(1, final_time)
        return range_time

    def set_anthropometry(self, mass, height, age):
        """Set subject's weight, height, age"""
        self.scale_tool.setSubjectMass(mass)
        self.scale_tool.setSubjectHeight(height)
        self.scale_tool.setSubjectAge(age)

    def run_model_scaler(self, mass):
        """Execute model scaling"""
        model_scaler = self.scale_tool.getModelScaler()
        model_scaler.setApply(True)
        model_scaler.setMarkerFileName(self.static_path)
        model_scaler.setTimeRange(self.time_range)
        model_scaler.setPreserveMassDist(True)
        model_scaler.setOutputModelFileName(self.model_output)
        model_scaler.setOutputScaleFileName(
            self.xml_output.replace(".xml", "_scaling_factor.xml")
        )

        model_scaler.processModel(self.model, "", mass)

        # 4.5 API does not provide getRMSMarkerError() functions, so no longer calling them

        print("[INFO] Model scaling (ModelScaler) completed.")

    def run_marker_placer(self):
        """Execute marker placement (MarkerPlacer)"""
        scaled_model = osim.Model(self.model_output)

        marker_placer = self.scale_tool.getMarkerPlacer()
        marker_placer.setApply(True)
        marker_placer.setTimeRange(self.time_range)
        marker_placer.setStaticPoseFileName(self.static_path)
        marker_placer.setOutputModelFileName(self.model_with_markers_output)
        marker_placer.setMaxMarkerMovement(-1)

        marker_placer.processModel(scaled_model)

        print("[INFO] Marker placement (MarkerPlacer) completed.")

        # Save final model & updated configuration
        scaled_model.printToXML(self.model_output)
        self.scale_tool.printToXML(self.xml_output)

    def compute_marker_error(self, model_path, trc_path, label="Marker Error"):
        """
        Manually calculate RMS / Max error between given model (model_path) and static marker file (trc_path).

        1) Read and average .trc file to get average position of each marker in experimental coordinate system (default entire time period).
        2) Load model, initSystem(), use getLocationInGround() for each Marker to get coordinates in world coordinate system.
        3) Compare differences between the two to get RMS and Max distance.
        4) Print results.

        Note: If experimental data and model units are inconsistent (e.g. mm vs m), unit conversion needs to be done manually.
        """
        print(f"\n[INFO] Starting to calculate {label} ...")

        # 1. Get "average coordinates" from .trc file
        avg_positions = self._get_average_marker_positions(trc_path)

        # 2. Load model and initialize
        model = osim.Model(model_path)
        state = model.initSystem()
        marker_set = model.updMarkerSet()

        # 3. Traverse each Marker in the model and calculate error with experimental average coordinates
        distances = []
        for i in range(marker_set.getSize()):
            mk = marker_set.get(i)
            mkName = mk.getName()
            # Only calculate markers with the same name that exist in trc file
            if mkName in avg_positions:
                # Get coordinates of this marker in ground coordinate system from model
                loc = mk.getLocationInGround(state)  # SimTK.Vec3
                dx = loc.get(0) - avg_positions[mkName][0]
                dy = loc.get(1) - avg_positions[mkName][1]
                dz = loc.get(2) - avg_positions[mkName][2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                distances.append(dist)

        if len(distances) == 0:
            print(f"[WARN] No comparable markers, unable to calculate {label}.")
            return

        rms_error = math.sqrt(sum(d * d for d in distances) / len(distances))
        max_error = max(distances)

        print(f"[RESULT] {label} - Number of markers: {len(distances)}")
        print(f"         RMS Marker Error: {rms_error:.4f}")
        print(f"         Max Marker Error: {max_error:.4f}")

    def _get_average_marker_positions(self, trc_file):
        """
        Read .trc file (MarkerData), average all frames, return {markerName: [x_avg, y_avg, z_avg]} dictionary.

        If your .trc is in millimeters and model is in meters, please multiply by 0.001 for unit conversion manually.
        """
        md = osim.MarkerData(trc_file)

        # May need to check units, e.g. when md.getUnits() == "mm" do conversion, not handled in this example
        markerNames = md.getMarkerNames()
        numMarkers = markerNames.getSize()
        numFrames = md.getNumFrames()

        # Initialize accumulators
        sums = {}
        for i in range(numMarkers):
            name = markerNames.get(i)
            sums[name] = [0.0, 0.0, 0.0]

        # Accumulate (x, y, z) for each frame
        for f in range(numFrames):
            frame = md.getFrame(f)
            for m in range(numMarkers):
                name = markerNames.get(m)
                pos = frame.getMarker(m)  # SimTK.Vec3
                sums[name][0] += pos.get(0)
                sums[name][1] += pos.get(1)
                sums[name][2] += pos.get(2)

        # Take average
        avg_positions = {}
        for i in range(numMarkers):
            name = markerNames.get(i)
            avg_positions[name] = [
                sums[name][0] / numFrames,
                sums[name][1] / numFrames,
                sums[name][2] / numFrames
            ]
        return avg_positions

    def add_unused_markers(self):
        """Add back unused Markers from original model to final model (optional)."""
        with_unused = osim.Model(self.model_output)
        without_unused = osim.Model(self.model_with_markers_output)

        with_unused_set = with_unused.getMarkerSet()
        without_unused_set = without_unused.getMarkerSet()

        all_names = {with_unused_set.get(i).getName() for i in range(with_unused.getNumMarkers())}
        used_names = {without_unused_set.get(i).getName() for i in range(without_unused.getNumMarkers())}

        for name in all_names - used_names:
            marker = with_unused_set.get(name).clone()
            without_unused.addMarker(marker)

        without_unused.printToXML(self.model_with_markers_output)

    def combine_models(self, model_to_add):
        """Combine other models into current scaled model (optional)."""
        base = osim.Model(self.model_output)
        add = osim.Model(model_to_add)

        for body in add.getBodySet():
            base.addBody(body.clone())
        for joint in add.getJointSet():
            base.addJoint(joint.clone())
        for ctrl in add.getControllerSet():
            base.addControl(ctrl.clone())
        for cst in add.getConstraintSet():
            base.addConstraint(cst.clone())
        for marker in add.getMarkerSet():
            base.addMarker(marker.clone())

        base.initSystem()

        # Can also add geometry search path again
        orig_geom_dir = os.path.join(os.path.dirname(os.path.abspath(self.model_input)), "Geometry")
        osim.ModelVisualizer.addDirToGeometrySearchPaths(orig_geom_dir)

        base.printToXML(self.model_with_markers_output)
        print(f"Combined {model_to_add} into {self.model_with_markers_output}")


def main():
    print("Starting OpenSim model scaling example...\n")

    model_input_path = r"C:\temporary_file\BG_klinik\newPipeline\config\models\ms_arm_and_hand-main\AAH Model\RightArmAndHand.osim"
    xml_input_path = r"C:\temporary_file\BG_klinik\newPipeline\config\models\ms_arm_and_hand-main\AAH Model\scaling_hansANDarms.xml"
    static_path = r"C:\temporary_file\BG_klinik\newPipeline\data\processed\Static 1_gm.trc"
    output_dir = r"C:\temporary_file\BG_klinik\newPipeline\data\processed"

    model_output = os.path.join(output_dir, "RightArmAndHand_scaled.osim")
    xml_output = os.path.join(output_dir, "scaling_hansANDarms_scaled.xml")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get height, weight, age from configuration
    conf = Conf(output_dir)
    try:
        mass = conf.get_conf_field("test_participant", ["mass"])
        height = conf.get_conf_field("test_participant", ["height"])
        age = conf.get_conf_field("test_participant", ["age"])
    except ValueError:
        mass, height, age = 70.0, 175.0, 30

    print("=== Scaling Configuration ===")
    print(f"Model file      : {model_input_path}")
    print(f"Config file     : {xml_input_path}")
    print(f"Static markers  : {static_path}")
    print(f"Output model    : {model_output}")
    print(f"Output config   : {xml_output}")
    print(f"Weight          : {mass} kg")
    print(f"Height          : {height} cm")
    print(f"Age             : {age} years\n")

    try:
        # If your scaling.xml uses meters, then pass height/100.0
        # If it uses cm, use height; if mm, use height*10 (example).
        Scale(
            model_input=model_input_path,
            model_output=model_output,
            xml_input=xml_input_path,
            xml_output=xml_output,
            static_path=static_path,
            mass_input=mass,
            height=height * 10,  # Please adjust according to actual units
            age=age,
            remove_unused=False
        )
        print("\nModel scaling & manual error calculation completed!")
    except Exception as e:
        print(f"Error during model scaling process: {e}")
        import traceback;
        traceback.print_exc()

    print("\nExample program execution completed")


if __name__ == "__main__":
    main()
