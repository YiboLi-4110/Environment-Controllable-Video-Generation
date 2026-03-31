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

def spawn_random_spheres():
    """随机生成球体数量、位置、尺寸以及随机错峰下落时间"""
    count = random.randint(7, 15) # 7到15个球
    # 至少5个球从第1帧开始下落
    n_from_frame_one = random.randint(5, min(7, count))
    early_indices = list(range(count))
    random.shuffle(early_indices)
    early_set = set(early_indices[:n_from_frame_one])

    sphere_data = []
    
    for i in range(count):
        # 1. 随机尺寸
        radius = random.uniform(0.2, 0.8)

        # 2. 随机初始位置 (基于你的相机位景，在上方 y=0 附近)
        # 假设相机在 Z=4，看向原点。球体在 Z=8~12 随机掉落
        loc_x = random.uniform(-9.0, 9.0)
        loc_y = random.uniform(-1.5, 1.5) 
        loc_z = random.uniform(2.0, 10.0)
        
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

    # 根据sample_id和重力大小设置输出目录名称
    output_dir_name = f"sample_{sample_id}_g_{metadata['gravity_z']:.2f}"
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
    spheres = spawn_random_spheres()
    
    metadata = {
        "sample_id": sample_id,
        "gravity_z": g,
        "hdri_file": hdri,
        # "spheres": spheres,
        "fps": FPS,
        "total_frames": TOTAL_FRAMES,
        "frame_ext": FRAME_EXT,
        # "jpeg_quality": JPEG_QUALITY,
    }
    
    bake_and_render(sample_id, render_root, metadata)

if __name__ == "__main__":
    main()