import bpy
import random
import os
import json
import math
import glob
from mathutils import Vector

# 与 falling_render.sh / 元数据一致：81 帧序列
TOTAL_FRAMES = 81
FPS = 16

# 序列帧为 JPEG；quality 约 95 兼顾体积与块效应
JPEG_QUALITY = 95
FRAME_EXT = "jpg"

TEXTURE_PATH = ".cache/football_textures"
# 与 rolling_balls 的 ground_textures 结构一致：每个子目录下含 textures/，内含 diff_4k / nor_gl_4k 等
GROUND_TEXTURE_PATH = ".cache/ground_textures"

# --- 刚体物理随机范围（合成数据 / 仿真文献常见做法：在「可辨识多样性」与「物理可信」之间折中）---
# 地面摩擦：Blender 中 friction∈[0,1]；工业与机器人 sim-to-real 常取 0.3~0.9 表示干燥硬质接触，
# 避免接近 0 的极端打滑（除非刻意模拟冰面），也避免全 1 导致数值过黏。
GROUND_FRICTION_RANGE = (0.35, 0.90)
# 地面恢复系数：真实路面/地垫对宏观碰撞的 e 多数 <0.5；0~0.95 过宽易产生不自然的超高弹。
GROUND_RESTITUTION_RANGE = (0.0, 0.50)
# 小球摩擦：与另一物体摩擦共同决定切向滑动；随机化可区分「表面材质」而不只依赖地面。
SPHERE_FRICTION_RANGE = (0.25, 0.85)
# 小球恢复系数：橡胶/塑料球常见中等反弹；与地面对共同决定反弹高度。
SPHERE_RESTITUTION_RANGE = (0.20, 0.72)

# 静止参考球数量（「几个」）
REF_AIR_SPHERE_COUNT_RANGE = (2, 4)
REF_GROUND_SPHERE_COUNT_RANGE = (2, 4)
# 放置时与已有球心的最小间隙（略大于 0 避免初始穿透）
SPHERE_PLACEMENT_MARGIN = 0.15


def _world_top_z_mesh(plane):
    """水平或近似水平地面：取包围盒世界坐标最大 Z 作为球心放置参考高度。"""
    zs = []
    for i in range(8):
        c = Vector(plane.bound_box[i])
        zs.append((plane.matrix_world @ c).z)
    return max(zs)


def _try_place_sphere(radius, z_min, z_max, x_bounds, y_bounds, occupied, max_attempts=120):
    """
    在轴对齐盒内随机放置球心，使与 occupied 中每个球 (pos, r) 满足 |c-c'| >= r+r'+margin。
    失败返回 None。
    """
    for _ in range(max_attempts):
        x = random.uniform(x_bounds[0], x_bounds[1])
        y = random.uniform(y_bounds[0], y_bounds[1])
        z = random.uniform(z_min, z_max)
        pos = Vector((x, y, z))
        ok = True
        for o in occupied:
            op = Vector(o["pos"])
            need = o["radius"] + radius + SPHERE_PLACEMENT_MARGIN
            if (pos - op).length < need:
                ok = False
                break
        if ok:
            return (x, y, z)
    return None


def get_ground_plane():
    """优先使用名为 Plane 的网格（与 falling.blend 一致），否则回退到名称匹配的平面。"""
    obj = bpy.data.objects.get("Plane")
    if obj and obj.type == "MESH":
        return obj
    for o in bpy.data.objects:
        if o.type == "MESH" and (o.name.startswith("Plane") or "plane" in o.name.lower()):
            return o
    return None


def apply_ground_textures(plane):
    """
    从 GROUND_TEXTURE_PATH 随机选一地面材质包，为平面构建 Principled + 贴图节点。
    目录结构与 force-prompting rolling_balls 相同：{pack}/textures/ 下按文件名匹配 diff_4k、nor_gl_4k 等。
    返回用于元数据的材质包目录名（去掉 .blend 后缀）；失败时返回 None。
    """
    if not plane or plane.type != "MESH":
        print("Warning: Invalid plane object for ground textures")
        return None

    ground_types = []
    try:
        ground_types = [
            d
            for d in os.listdir(GROUND_TEXTURE_PATH)
            if os.path.isdir(os.path.join(GROUND_TEXTURE_PATH, d))
        ]
    except Exception as e:
        print(f"Error reading ground texture directories: {e}")
        return None

    if not ground_types:
        print(f"No ground texture directories found in {GROUND_TEXTURE_PATH}")
        return None

    selected_ground = random.choice(ground_types)
    texture_path = os.path.join(GROUND_TEXTURE_PATH, selected_ground, "textures")
    meta_name = selected_ground.split(".blend")[0]

    print(f"Selected ground texture pack: {selected_ground}")
    if not os.path.isdir(texture_path):
        print(f"Warning: Expected textures folder missing: {texture_path}")
        return None

    diffuse_path = None
    normal_path = None
    roughness_path = None
    displacement_path = None

    try:
        all_files = os.listdir(texture_path)
    except Exception as e:
        print(f"Error listing texture folder {texture_path}: {e}")
        return None

    for file in all_files:
        file_lower = file.lower()
        if "diff_4k" in file_lower and file_lower.endswith(".jpg"):
            diffuse_path = os.path.join(texture_path, file)
        elif "nor_gl_4k" in file_lower and file_lower.endswith(".exr"):
            normal_path = os.path.join(texture_path, file)
        elif "rough_4k" in file_lower:
            if file_lower.endswith(".jpg") or file_lower.endswith(".exr"):
                roughness_path = os.path.join(texture_path, file)
        elif "disp_4k" in file_lower:
            if file_lower.endswith(".jpg") or file_lower.endswith(".png"):
                displacement_path = os.path.join(texture_path, file)

    print(f"Found diffuse: {os.path.basename(diffuse_path) if diffuse_path else 'None'}")
    print(f"Found normal: {os.path.basename(normal_path) if normal_path else 'None'}")
    print(f"Found roughness: {os.path.basename(roughness_path) if roughness_path else 'None'}")
    print(f"Found displacement: {os.path.basename(displacement_path) if displacement_path else 'None'}")

    if len(plane.material_slots) == 0:
        mat = bpy.data.materials.new(name="Ground_Material")
        plane.data.materials.append(mat)
    else:
        mat = plane.material_slots[0].material
        if not mat:
            mat = bpy.data.materials.new(name="Ground_Material")
            plane.material_slots[0].material = mat

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.location = (0, 0)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = (-800, 0)
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.location = (-600, 0)
    mapping.inputs["Scale"].default_value[0] = 5.0
    mapping.inputs["Scale"].default_value[1] = 5.0
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    if diffuse_path and os.path.exists(diffuse_path):
        tex_diffuse = nodes.new(type="ShaderNodeTexImage")
        tex_diffuse.location = (-400, 200)
        tex_diffuse.image = bpy.data.images.load(diffuse_path)
        links.new(mapping.outputs["Vector"], tex_diffuse.inputs["Vector"])
        links.new(tex_diffuse.outputs["Color"], principled.inputs["Base Color"])
        print(f"Applied ground diffuse texture: {diffuse_path}")
    else:
        print("Warning: Ground diffuse texture not found")

    if normal_path and os.path.exists(normal_path):
        tex_normal = nodes.new(type="ShaderNodeTexImage")
        tex_normal.location = (-400, 0)
        tex_normal.image = bpy.data.images.load(normal_path)
        normal_map = nodes.new(type="ShaderNodeNormalMap")
        normal_map.location = (-200, 0)
        links.new(mapping.outputs["Vector"], tex_normal.inputs["Vector"])
        links.new(tex_normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
        print(f"Applied ground normal texture: {normal_path}")
    else:
        print("Warning: Ground normal texture not found")

    if roughness_path and os.path.exists(roughness_path):
        tex_roughness = nodes.new(type="ShaderNodeTexImage")
        tex_roughness.location = (-400, -200)
        tex_roughness.image = bpy.data.images.load(roughness_path)
        links.new(mapping.outputs["Vector"], tex_roughness.inputs["Vector"])
        links.new(tex_roughness.outputs["Color"], principled.inputs["Roughness"])
        print(f"Applied ground roughness texture: {roughness_path}")
    else:
        print("Warning: Ground roughness texture not found")

    if displacement_path and os.path.exists(displacement_path):
        tex_disp = nodes.new(type="ShaderNodeTexImage")
        tex_disp.location = (-400, -400)
        tex_disp.image = bpy.data.images.load(displacement_path)
        disp_node = nodes.new(type="ShaderNodeDisplacement")
        disp_node.location = (-200, -400)
        disp_node.inputs["Scale"].default_value = 0.05
        links.new(mapping.outputs["Vector"], tex_disp.inputs["Vector"])
        links.new(tex_disp.outputs["Color"], disp_node.inputs["Height"])
        links.new(disp_node.outputs["Displacement"], output.inputs["Displacement"])

        if plane.modifiers.get("Subdivision") is None:
            subdiv = plane.modifiers.new(name="Subdivision", type="SUBSURF")
            subdiv.levels = 2
            subdiv.render_levels = 2

        # Blender 4.x Cycles：位移与凹凸；若属性不存在则跳过以免版本差异报错
        if getattr(mat, "cycles", None) is not None:
            mat.cycles.displacement_method = "BOTH"
        print(f"Applied ground displacement texture: {displacement_path}")
    else:
        print("Warning: Ground displacement texture not found")

    return meta_name


def randomize_plane_rigid_body(plane):
    """
    保证地面为 Passive 刚体并用于碰撞；随机化恢复系数 restitution 与摩擦系数 friction。
    RNA 属性名始终为 restitution（物理学术语）；界面中可能显示为「Bounciness」等与用户友好的标签。
    friction：Bullet 中常用 0~1，越大切向阻力越强。
    """
    if not plane or plane.type != "MESH":
        return None

    bpy.ops.object.select_all(action="DESELECT")
    plane.select_set(True)
    bpy.context.view_layer.objects.active = plane

    if not plane.rigid_body:
        bpy.ops.rigidbody.object_add(type="PASSIVE")
    else:
        plane.rigid_body.type = "PASSIVE"

    plane.rigid_body.collision_shape = "MESH"
    friction = random.uniform(*GROUND_FRICTION_RANGE)
    plane.rigid_body.friction = friction
    restitution = random.uniform(*GROUND_RESTITUTION_RANGE)
    plane.rigid_body.restitution = restitution
    print(f"Ground rigid body: PASSIVE, restitution={restitution:.4f}, friction={friction:.4f}")
    return restitution, friction


def smooth_ball_surface(obj):
    """让球体表面更平滑：平滑法线 + 轻量细分，兼顾质量与渲染开销。"""
    if not obj or obj.type != "MESH":
        return

    # Blender 4.4: 显式将每个面设为平滑着色，避免出现分面感。
    for poly in obj.data.polygons:
        poly.use_smooth = True

    # 增加轻量细分，改善球体轮廓与高光过渡；不宜过高以控制渲染成本。
    subdiv = obj.modifiers.get("BallSmoothSubdiv")
    if subdiv is None:
        subdiv = obj.modifiers.new(name="BallSmoothSubdiv", type="SUBSURF")
    subdiv.levels = 1
    subdiv.render_levels = 2
    subdiv.subdivision_type = "CATMULL_CLARK"


def apply_ball_textures(obj, texture_folder=None):
    """从 TEXTURE_PATH 随机选取子目录，给球体赋予 pattern/normal/rough 纹理材质。"""
    if not obj:
        print("Warning: Target object not found")
        return None

    if texture_folder is None:
        subfolders = [f for f in os.listdir(TEXTURE_PATH) if os.path.isdir(os.path.join(TEXTURE_PATH, f))]
        if not subfolders:
            print(f"Warning: No texture subfolders found in {TEXTURE_PATH}")
            return None
        texture_folder = random.choice(subfolders)

    pattern_path = os.path.join(TEXTURE_PATH, texture_folder)

    def find_texture(name_base):
        matches = glob.glob(os.path.join(pattern_path, f"{name_base}.*"))
        return matches[0] if matches else None

    diffuse_path = find_texture("pattern")
    normal_path = find_texture("normal")
    roughness_path = find_texture("rough")

    mat = bpy.data.materials.new(name=f"Sphere_Material_{obj.name}_{texture_folder}")
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)
    # 降低基础粗糙度，避免球体整体“发灰发糙”，仍保留贴图细节。
    principled.inputs['Roughness'].default_value = 0.22
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    if diffuse_path and os.path.exists(diffuse_path):
        tex_diffuse = nodes.new(type='ShaderNodeTexImage')
        tex_diffuse.location = (-300, 100)
        tex_diffuse.image = bpy.data.images.load(diffuse_path)
        links.new(tex_diffuse.outputs['Color'], principled.inputs['Base Color'])
        print(f"Applied diffuse texture: {diffuse_path}")
    else:
        print(f"Warning: No diffuse texture found in {pattern_path}")

    if normal_path and os.path.exists(normal_path):
        tex_normal = nodes.new(type='ShaderNodeTexImage')
        tex_normal.location = (-300, -100)
        tex_normal.image = bpy.data.images.load(normal_path)
        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (-100, -100)
        # 将法线强度限制在中低范围，保留纹理但减少“颗粒/坑洼感”。
        normal_map.inputs['Strength'].default_value = random.uniform(0.35, 0.70)
        links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        print(f"Applied normal texture: {normal_path}")
    else:
        print(f"Warning: No normal texture found in {pattern_path}")

    if roughness_path and os.path.exists(roughness_path):
        tex_roughness = nodes.new(type='ShaderNodeTexImage')
        tex_roughness.location = (-300, -300)
        tex_roughness.image = bpy.data.images.load(roughness_path)

        # 将 roughness 贴图影响缩放到较低范围：保留明暗变化，但不过度粗糙。
        rough_to_bw = nodes.new(type='ShaderNodeRGBToBW')
        rough_to_bw.location = (-130, -300)
        rough_scale = nodes.new(type='ShaderNodeMath')
        rough_scale.location = (60, -300)
        rough_scale.operation = 'MULTIPLY'
        rough_scale.inputs[1].default_value = random.uniform(0.40, 0.75)

        links.new(tex_roughness.outputs['Color'], rough_to_bw.inputs['Color'])
        links.new(rough_to_bw.outputs['Val'], rough_scale.inputs[0])
        links.new(rough_scale.outputs['Value'], principled.inputs['Roughness'])
        print(f"Applied roughness texture: {roughness_path}")
    else:
        print(f"Warning: No roughness texture found in {pattern_path}")

    return texture_folder

def _set_fcurve_constant_interpolation(obj, data_paths):
    """将指定属性的关键帧插值设为 CONSTANT，避免布尔/离散状态在帧间被平滑插值导致刚体求值闪烁。"""
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    path_set = set(data_paths)
    for fc in ad.action.fcurves:
        if fc.data_path in path_set:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"

def setup_gpu_cycles():
    """针对 4090 和 Blender 4.4 的渲染优化"""
    scene = bpy.context.scene

    # 强制将物理引擎与渲染输出的帧率统一
    scene.render.fps = FPS
    
    # 确保刚体世界的速度倍率是绝对的 1.0 (现实时间)
    if scene.rigidbody_world:
        scene.rigidbody_world.time_scale = 1.0

    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.get_devices()
    
    # 启用 CUDA 或 OPTIX
    device_type = 'OPTIX' if 'OPTIX' in [d.type for d in prefs.devices] else 'CUDA'
    prefs.compute_device_type = device_type
    
    for dev in prefs.devices:
        dev.use = (dev.type in {'CUDA', 'OPTIX'})
    
    scene.cycles.device = 'GPU'
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 7  # 匹配 32核/4卡 分配

    # 动画序列：复用场景数据，减少每帧准备开销（一般不改变画质）
    scene.render.use_persistent_data = True

    # 响应你提到的采样问题：设置 Render 采样
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True

    # 自适应采样：平坦区域提前结束，配合降噪观感接近固定采样上限
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_min_samples = 48
    scene.cycles.adaptive_threshold = 0.02

def randomize_gravity():
    """随机化重力大小 (First Principles: 物理一致性)"""
    # 随机重力范围：从月球(1.6)到地球(9.8)再到超重(20.0)
    # g_value = 9.81 # blender默认地球重力
    g_value = random.uniform(1.0, 20.0)
    bpy.context.scene.gravity[2] = -g_value
    return g_value

def randomize_hdri(hdri_path_root):
    """随机环境贴图"""
    if not os.path.exists(hdri_path_root):
        print(f"Error: HDRI path {hdri_path_root} not found")
        return "default"
    
    hdri_files = [f for f in os.listdir(hdri_path_root) if f.endswith(('.hdr', '.exr'))]
    selected_hdri = random.choice(hdri_files)
    
    world = bpy.context.scene.world
    world.use_nodes = True
    en_node = world.node_tree.nodes.get("Environment Texture") 
    # 注意：如果模板里没这个节点，脚本需新建（见之前建议）
    if en_node:
        img = bpy.data.images.load(os.path.join(hdri_path_root, selected_hdri))
        en_node.image = img
    return selected_hdri

def setup_supplementary_key_light():
    """在相机上方附近增加一束与 HDRI 配合的平行光（Sun），能量随机：0 为纯环境光，越大越亮。"""
    scene = bpy.context.scene
    cam = scene.camera
    if not cam:
        print("Warning: No scene camera; skip supplementary key light")
        return 0.0

    name = "Supplementary_Key_Sun"
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    # Cycles 中 Sun 仅由旋转决定方向；能量 0 等价于仅 HDRI
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 0.0))
    sun = bpy.context.active_object
    sun.name = name
    sun.data.color = (1.0, 0.98, 0.95)
    # 略偏暖，与常见 HDRI 协调；能量范围：0（纯自然光）~ 约 6（明显提亮但不易过曝）
    energy = random.uniform(50.0, 70.0)
    sun.data.energy = energy

    # 光线从「相机略上方」指向场景中心，照亮全景顶侧
    cam_loc = cam.matrix_world.translation
    light_origin = cam_loc + Vector((0.0, 0.0, 2.5))
    target = Vector((0.0, 0.0, 0.0))
    emit_dir = (target - light_origin).normalized()
    sun.rotation_euler = emit_dir.to_track_quat("-Z", "Y").to_euler()

    return energy


def spawn_reference_spheres(plane, occupied):
    """
    静止参考物：
    - 空中：Passive 刚体，不参与动力学但参与碰撞，位置固定。
    - 地面：Active 刚体，初始静止在平面之上；被下落球撞击后可滚动（Passive 被撞不会动）。
    occupied：已占位球心列表，本函数会追加，供下落球避让。
    """
    air_meta = []
    ground_meta = []
    n_air = random.randint(*REF_AIR_SPHERE_COUNT_RANGE)
    n_ground = random.randint(*REF_GROUND_SPHERE_COUNT_RANGE) if plane else 0

    top_z = _world_top_z_mesh(plane) if plane else 0.0

    air_idx = 0
    for _ in range(n_air):
        r = random.uniform(0.18, 0.55)
        pos = _try_place_sphere(r, 3.2, 9.5, (-8.5, 8.5), (-1.4, 1.4), occupied)
        if pos is None:
            print("Warning: could not place an air reference sphere (spacing); skip one")
            continue
        loc_x, loc_y, loc_z = pos
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(loc_x, loc_y, loc_z))
        obj = bpy.context.active_object
        obj.name = f"Ref_Sphere_Air_{air_idx}"
        air_idx += 1
        smooth_ball_surface(obj)
        tf = apply_ball_textures(obj)
        bpy.ops.rigidbody.object_add(type="PASSIVE")
        obj.rigid_body.collision_shape = "SPHERE"
        sf = random.uniform(*SPHERE_FRICTION_RANGE)
        sr = random.uniform(*SPHERE_RESTITUTION_RANGE)
        obj.rigid_body.friction = sf
        obj.rigid_body.restitution = sr
        occupied.append({"pos": (loc_x, loc_y, loc_z), "radius": r})
        air_meta.append(
            {
                "name": obj.name,
                "radius": r,
                "initial_pos": [loc_x, loc_y, loc_z],
                "rigid_body": "PASSIVE",
                "texture_folder": tf,
                "friction": sf,
                "restitution": sr,
            }
        )

    ground_idx = 0
    for _ in range(n_ground):
        r = random.uniform(0.18, 0.55)
        zc = top_z + r + 0.002
        pos = _try_place_sphere(r, zc, zc, (-7.5, 7.5), (-1.4, 1.4), occupied)
        if pos is None:
            print("Warning: could not place a ground reference sphere (spacing); skip one")
            continue
        loc_x, loc_y, loc_z = pos
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(loc_x, loc_y, loc_z))
        obj = bpy.context.active_object
        obj.name = f"Ref_Sphere_Ground_{ground_idx}"
        ground_idx += 1
        smooth_ball_surface(obj)
        tf = apply_ball_textures(obj)
        bpy.ops.rigidbody.object_add(type="ACTIVE")
        obj.rigid_body.type = "ACTIVE"
        obj.rigid_body.collision_shape = "SPHERE"
        obj.rigid_body.mass = 1.0
        sf = random.uniform(*SPHERE_FRICTION_RANGE)
        sr = random.uniform(*SPHERE_RESTITUTION_RANGE)
        obj.rigid_body.friction = sf
        obj.rigid_body.restitution = sr
        occupied.append({"pos": (loc_x, loc_y, loc_z), "radius": r})
        ground_meta.append(
            {
                "name": obj.name,
                "radius": r,
                "initial_pos": [loc_x, loc_y, loc_z],
                "rigid_body": "ACTIVE",
                "texture_folder": tf,
                "friction": sf,
                "restitution": sr,
            }
        )

    return {"air": air_meta, "ground": ground_meta}


def spawn_random_spheres(occupied=None):
    """随机生成球体数量、位置、尺寸以及随机错峰下落时间；occupied 为已占位球列表以避免初始重叠。"""
    if occupied is None:
        occupied = []
    count = random.randint(5, 12) # 5到12个球
    # 至少3个球从第1帧开始下落
    n_from_frame_one = random.randint(3, 5)
    early_indices = list(range(count))
    random.shuffle(early_indices)
    early_set = set(early_indices[:n_from_frame_one])

    sphere_data = []
    
    for i in range(count):
        # 1. 随机尺寸
        radius = random.uniform(0.2, 0.8)

        # 2. 随机初始位置，与参考球避让（失败则回退为完全随机）
        pos = _try_place_sphere(radius, 5.0, 10.0, (-9.0, 9.0), (-1.5, 1.5), occupied)
        if pos is None:
            loc_x = random.uniform(-9.0, 9.0)
            loc_y = random.uniform(-1.5, 1.5)
            loc_z = random.uniform(5.0, 10.0)
        else:
            loc_x, loc_y, loc_z = pos
        occupied.append({"pos": (loc_x, loc_y, loc_z), "radius": radius})

        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(loc_x, loc_y, loc_z))
        obj = bpy.context.active_object
        obj.name = f"Falling_Sphere_{i}"
        smooth_ball_surface(obj)

        # 3. 赋予纹理
        texture_folder = apply_ball_textures(obj)

        # 4. 赋予刚体属性
        bpy.ops.rigidbody.object_add(type='ACTIVE')
        obj.rigid_body.type = 'ACTIVE'
        obj.rigid_body.collision_shape = 'SPHERE'
        obj.rigid_body.mass = 1.0
        sphere_friction = random.uniform(*SPHERE_FRICTION_RANGE)
        obj.rigid_body.friction = sphere_friction
        sphere_restitution = random.uniform(*SPHERE_RESTITUTION_RANGE)
        obj.rigid_body.restitution = sphere_restitution
        
        # 5. 起始帧：early_set 内固定第 1 帧；其余在 2~40 帧错峰（保证至少 3 个从首帧下落）
        if i in early_set:
            start_fall_frame = 1
        else:
            start_fall_frame = random.randint(2, 40)

        # 错峰下落：运动学阶段必须显式锁定 location/rotation，且 kinematic 关键帧须为 CONSTANT 插值。
        # 否则 Blender 会对 rigid_body.kinematic 做默认（贝塞尔/线性）插值，布尔在帧间变成“中间值”，
        # 烘焙/渲染时会在动力学与固定初始位姿之间抖动，表现为“已下落又瞬移回起点”的闪现。
        if start_fall_frame > 1:
            obj.rigid_body.kinematic = True
            obj.keyframe_insert(data_path="rigid_body.kinematic", frame=1)
            obj.keyframe_insert(data_path="location", frame=1)
            obj.keyframe_insert(data_path="rotation_euler", frame=1)
            if start_fall_frame > 2:
                obj.keyframe_insert(data_path="rigid_body.kinematic", frame=start_fall_frame - 1)
                obj.keyframe_insert(data_path="location", frame=start_fall_frame - 1)
                obj.keyframe_insert(data_path="rotation_euler", frame=start_fall_frame - 1)

        obj.rigid_body.kinematic = False
        obj.keyframe_insert(data_path="rigid_body.kinematic", frame=start_fall_frame)

        _set_fcurve_constant_interpolation(
            obj,
            ("rigid_body.kinematic", "location", "rotation_euler"),
        )

        # 记录元数据，包括start_fall_frame
        sphere_data.append({
            "id": i,
            "radius": radius,
            "initial_pos": [loc_x, loc_y, loc_z],
            "texture_folder": texture_folder,
            "friction": sphere_friction,
            "restitution": sphere_restitution,
            "start_fall_frame": start_fall_frame  # 增加时间戳数据
        })
    return sphere_data

def bake_and_render(sample_id, render_dir, metadata):
    """执行物理烘焙并渲染序列帧（显式 render，不依赖 CLI 的 -a）"""
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES

    # 刚体 point_cache 默认帧范围常为 250，不收紧会导致 bake/渲染不同步
    rbw = scene.rigidbody_world
    if rbw and rbw.point_cache:
        rbw.point_cache.frame_start = 1
        rbw.point_cache.frame_end = TOTAL_FRAMES

    # 根据sample_id、重力大小、hdri背景图名称设置输出目录名称
    hdri_file_name = metadata['hdri_file'].split(".")[0]
    output_dir_name = f"sample_{sample_id}_g_{metadata['gravity_z']:.2f}_{hdri_file_name}"
    output_path = os.path.abspath(os.path.join(render_dir, output_dir_name))
    os.makedirs(output_path, exist_ok=True)

    print(f"Baking physics for sample {sample_id}...")
    bpy.ops.ptcache.free_bake_all()
    bpy.context.view_layer.update()
    bpy.ops.ptcache.bake_all(bake=True)

    # bake 会推进 frame_current；序列渲染应从第 1 帧开始
    scene.frame_set(1)
    bpy.context.view_layer.update()

    scene.render.filepath = os.path.join(output_path, "frame#")
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = metadata.get("jpeg_quality", JPEG_QUALITY)
    scene.render.use_file_extension = True

    with open(os.path.join(output_path, "params.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Rendering {TOTAL_FRAMES} frames to {output_path} ...")
    bpy.ops.render.render(animation=True)
    print(f"Done sample {sample_id}.")

def main():
    # 解析命令行参数 (由 Shell 脚本传入)
    import sys
    argv = sys.argv
    try:
        idx = argv.index("--")
        sample_id = argv[idx+2]
        render_root = argv[idx+4]
    except:
        sample_id = "test"
        render_root = "./output"

    setup_gpu_cycles()
    
    # 执行随机化
    g = randomize_gravity()
    hdri = randomize_hdri(".cache/HDRIs")

    ground_plane = get_ground_plane()
    ground_texture_name = None
    ground_restitution = None
    ground_friction = None
    if ground_plane:
        ground_texture_name = apply_ground_textures(ground_plane)
        rb_ground = randomize_plane_rigid_body(ground_plane)
        if rb_ground is not None:
            ground_restitution, ground_friction = rb_ground
    else:
        print("Warning: No ground plane (Plane) found; skip ground textures and plane rigid body")

    occupied = []
    reference_spheres = spawn_reference_spheres(ground_plane, occupied)
    spheres = spawn_random_spheres(occupied)

    metadata = {
        "sample_id": sample_id,
        "gravity_z": g,
        "hdri_file": hdri,
        "ground_texture": ground_texture_name,
        # "ground_restitution": ground_restitution,
        # "ground_friction": ground_friction,
        # "physics_sampling_ranges": {
        #     "ground_friction": list(GROUND_FRICTION_RANGE),
        #     "ground_restitution": list(GROUND_RESTITUTION_RANGE),
        #     "sphere_friction": list(SPHERE_FRICTION_RANGE),
        #     "sphere_restitution": list(SPHERE_RESTITUTION_RANGE),
        # },
        # "reference_spheres": reference_spheres,
        # "spheres": spheres,
        "fps": FPS,
        "total_frames": TOTAL_FRAMES,
        "frame_ext": FRAME_EXT,
        # "jpeg_quality": JPEG_QUALITY,
    }
    
    bake_and_render(sample_id, render_root, metadata)

if __name__ == "__main__":
    main()