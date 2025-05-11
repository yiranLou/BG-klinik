import opensim as osim
from pathlib import Path
import os
import sys
import math


class Conf:
    """简化的配置类，用于测试目的"""

    def __init__(self, project_path):
        self.project_path = Path(project_path)
        # 模拟配置数据
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
    Scale tool in pyosim - pyosim中的缩放工具。
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

        # 读取模型
        self.model = osim.Model(model_input)

        # 生成 ScaleTool 实例
        self.scale_tool = osim.ScaleTool(xml_input)
        self.set_anthropometry(mass_input, height, age)
        self.scale_tool.getGenericModelMaker().setModelFileName(model_input)

        # 设置时间范围（参考：静态标记数据）
        self.time_range = self.time_range_from_static()

        # 先添加几何文件搜索路径，避免出现找不到 .vtp 警告
        orig_geom_dir = os.path.join(os.path.dirname(os.path.abspath(model_input)), "Geometry")
        out_geom_dir = os.path.join(os.path.dirname(os.path.abspath(model_output)), "Geometry")
        osim.ModelVisualizer.addDirToGeometrySearchPaths(orig_geom_dir)
        osim.ModelVisualizer.addDirToGeometrySearchPaths(out_geom_dir)

        # 1. 运行模型缩放 (ModelScaler)
        self.run_model_scaler(mass_input)
        # 2. 手工计算一下缩放后的模型与静态标记的误差
        self.compute_marker_error(
            model_path=self.model_output,
            trc_path=self.static_path,
            label="模型缩放 (ModelScaler) 误差"
        )

        # 3. 标记放置 (MarkerPlacer)
        self.run_marker_placer()
        # 4. 手工计算一下放置标记后的模型与静态标记的误差
        self.compute_marker_error(
            model_path=self.model_with_markers_output,
            trc_path=self.static_path,
            label="标记放置 (MarkerPlacer) 误差"
        )

        # 5. 选择性：合并其他模型
        if add_model:
            self.combine_models(add_model)

        # 6. 选择性：把原模型里未被使用的 markers 加回去
        if not remove_unused:
            self.add_unused_markers()

    def time_range_from_static(self):
        """返回 [start_time, end_time]"""
        md = osim.MarkerData(self.static_path)
        initial_time = md.getStartFrameTime()
        final_time = md.getLastFrameTime()
        range_time = osim.ArrayDouble()
        range_time.set(0, initial_time)
        range_time.set(1, final_time)
        return range_time

    def set_anthropometry(self, mass, height, age):
        """设置被试者的体重、身高、年龄"""
        self.scale_tool.setSubjectMass(mass)
        self.scale_tool.setSubjectHeight(height)
        self.scale_tool.setSubjectAge(age)

    def run_model_scaler(self, mass):
        """执行模型缩放"""
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

        # 4.5 API 不提供 getRMSMarkerError() 等函数，故不再调用

        print("[INFO] 模型缩放 (ModelScaler) 已执行完毕.")

    def run_marker_placer(self):
        """执行标记放置 (MarkerPlacer)"""
        scaled_model = osim.Model(self.model_output)

        marker_placer = self.scale_tool.getMarkerPlacer()
        marker_placer.setApply(True)
        marker_placer.setTimeRange(self.time_range)
        marker_placer.setStaticPoseFileName(self.static_path)
        marker_placer.setOutputModelFileName(self.model_with_markers_output)
        marker_placer.setMaxMarkerMovement(-1)

        marker_placer.processModel(scaled_model)

        print("[INFO] 标记放置 (MarkerPlacer) 已执行完毕.")

        # 保存最终模型 & 更新后的配置
        scaled_model.printToXML(self.model_output)
        self.scale_tool.printToXML(self.xml_output)

    def compute_marker_error(self, model_path, trc_path, label="Marker误差"):
        """
        手工计算给定模型 (model_path) 与静态标记文件 (trc_path) 的 RMS / Max 误差。

        1) 读取并平均 .trc 文件，得到各marker在实验坐标系的平均位置(默认整个时段)。
        2) 加载模型、initSystem()，对每个 Marker 用 getLocationInGround() 获取在世界系的坐标。
        3) 比较两者差值，得到 RMS 和 Max 距离。
        4) 打印结果。

        注意：若实验数据和模型单位不一致(如 mm vs m)，需要自行做单位转换。
        """
        print(f"\n[INFO] 开始计算 {label} ...")

        # 1. 从 .trc 文件获取 “平均坐标”
        avg_positions = self._get_average_marker_positions(trc_path)

        # 2. 读取模型并初始化
        model = osim.Model(model_path)
        state = model.initSystem()
        marker_set = model.updMarkerSet()

        # 3. 遍历模型中的每个Marker，计算和实验平均坐标的误差
        distances = []
        for i in range(marker_set.getSize()):
            mk = marker_set.get(i)
            mkName = mk.getName()
            # 只计算在 trc 文件里存在的同名 marker
            if mkName in avg_positions:
                # 得到模型中该 marker 在 ground 的坐标
                loc = mk.getLocationInGround(state)  # SimTK.Vec3
                dx = loc.get(0) - avg_positions[mkName][0]
                dy = loc.get(1) - avg_positions[mkName][1]
                dz = loc.get(2) - avg_positions[mkName][2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                distances.append(dist)

        if len(distances) == 0:
            print(f"[WARN] 无可比对的标记，无法计算 {label}。")
            return

        rms_error = math.sqrt(sum(d * d for d in distances) / len(distances))
        max_error = max(distances)

        print(f"[RESULT] {label} - 标记数: {len(distances)}")
        print(f"         RMS Marker Error: {rms_error:.4f}")
        print(f"         Max Marker Error: {max_error:.4f}")

    def _get_average_marker_positions(self, trc_file):
        """
        读取 .trc 文件 (MarkerData)，对所有帧做平均，返回 {markerName: [x_avg, y_avg, z_avg]} 字典。

        如果你的 .trc 是毫米而模型是米，请自行乘以 0.001 做单位转换。
        """
        md = osim.MarkerData(trc_file)

        # 可能需要检查units，如 md.getUnits() == "mm" 时做转换，这里示例暂不处理
        markerNames = md.getMarkerNames()
        numMarkers = markerNames.getSize()
        numFrames = md.getNumFrames()

        # 初始化累加器
        sums = {}
        for i in range(numMarkers):
            name = markerNames.get(i)
            sums[name] = [0.0, 0.0, 0.0]

        # 累加每帧的 (x, y, z)
        for f in range(numFrames):
            frame = md.getFrame(f)
            for m in range(numMarkers):
                name = markerNames.get(m)
                pos = frame.getMarker(m)  # SimTK.Vec3
                sums[name][0] += pos.get(0)
                sums[name][1] += pos.get(1)
                sums[name][2] += pos.get(2)

        # 取平均
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
        """将原模型里未被使用的 Marker 加回最终模型(可选)。"""
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
        """将其它模型合并进当前 scaled 模型（可选）。"""
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

        # 也可以再加一次几何搜索路径
        orig_geom_dir = os.path.join(os.path.dirname(os.path.abspath(self.model_input)), "Geometry")
        osim.ModelVisualizer.addDirToGeometrySearchPaths(orig_geom_dir)

        base.printToXML(self.model_with_markers_output)
        print(f"已将 {model_to_add} 合并至 {self.model_with_markers_output}")


def main():
    print("开始运行OpenSim模型缩放示例...\n")

    model_input_path = r"C:\temporary_file\BG_klinik\newPipeline\config\models\ms_arm_and_hand-main\AAH Model\RightArmAndHand.osim"
    xml_input_path = r"C:\temporary_file\BG_klinik\newPipeline\config\models\ms_arm_and_hand-main\AAH Model\scaling_hansANDarms.xml"
    static_path = r"C:\temporary_file\BG_klinik\newPipeline\data\processed\Static 1_gm.trc"
    output_dir = r"C:\temporary_file\BG_klinik\newPipeline\data\processed"

    model_output = os.path.join(output_dir, "RightArmAndHand_scaled.osim")
    xml_output = os.path.join(output_dir, "scaling_hansANDarms_scaled.xml")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 从配置获取身高、体重、年龄
    conf = Conf(output_dir)
    try:
        mass = conf.get_conf_field("test_participant", ["mass"])
        height = conf.get_conf_field("test_participant", ["height"])
        age = conf.get_conf_field("test_participant", ["age"])
    except ValueError:
        mass, height, age = 70.0, 175.0, 30

    print("=== 缩放配置 ===")
    print(f"模型文件     : {model_input_path}")
    print(f"配置文件     : {xml_input_path}")
    print(f"静态标记文件 : {static_path}")
    print(f"输出模型     : {model_output}")
    print(f"输出配置     : {xml_output}")
    print(f"体重         : {mass} kg")
    print(f"身高         : {height} cm")
    print(f"年龄         : {age} 岁\n")

    try:
        # 如果你的 scaling.xml 里使用单位米，那么应传入 height/100.0
        # 如果它使用 cm，就用 height；若用 mm，就用 height*10 (示例)。
        Scale(
            model_input=model_input_path,
            model_output=model_output,
            xml_input=xml_input_path,
            xml_output=xml_output,
            static_path=static_path,
            mass_input=mass,
            height=height * 10,  # 请根据实际单位调整
            age=age,
            remove_unused=False
        )
        print("\n模型缩放&手工误差计算完成!")
    except Exception as e:
        print(f"模型缩放过程中出错: {e}")
        import traceback;
        traceback.print_exc()

    print("\n示例程序执行完毕")


if __name__ == "__main__":
    main()
