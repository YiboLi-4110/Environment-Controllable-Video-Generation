import bpy
import random
import os
import json
import math
import glob
from mathutils import Vector, Matrix

TOTAL_FRAMES = 81
FPS = 16
JPEG_QUALITY = 95
FRAME_EXT = "jpg"

TEXTURE_PATH = ".cache/football_textures"
GROUND_TEXTURE_PATH = ".cache/ground_textures"

# ──────────────── 相机可视范围（由用户提供） ────────────────
CAM_X_MIN, CAM_X_MAX = -5.8, 5.8
CAM_Z_MIN, CAM_Z_MAX = 0.0, 5.0

# ──────────────── 斜面 (Wedge) 参数范围 ────────────────
WEDGE_HEIGHT_RANGE = (2.0, 3.0)
WEDGE_BASE_RANGE = (1.8, 2.5)
WEDGE_WIDTH_RANGE = (4.0, 5.2)
LOCX_RANGE = (4.0, 4.4)
LOCY_RANGE = (7.8, 13.0)

# ──────────────── 物理参数 ────────────────
GROUND_FRICTION_RANGE = (0.35, 0.90)
GROUND_RESTITUTION_RANGE = (0.0, 0.50)
WEDGE_FRICTION_RANGE = (0.30, 0.40)
WEDGE_RESTITUTION_RANGE = (0.05, 0.40)
SLIDER_FRICTION_RANGE = (0.20, 0.30)
SLIDER_RESTITUTION_RANGE = (0.05, 0.30)
ROLLER_FRICTION_RANGE = (0.15, 0.65)
ROLLER_RESTITUTION_RANGE = (0.10, 0.50)

SLIDER_SCALE_RANGE = (1.0, 1.5)
ROLLER_RADIUS_RANGE = (0.7, 1.0)

# 滑块与滚球共用材料密度 ρ（kg / Blender 长度单位³）；m_立方体 = ρ·s³，m_球 = ρ·(4/3)πr³
OBJECT_MATERIAL_DENSITY = 4.5


def _world_top_z_mesh(obj):
    zs = []
    for i in range(8):
        c = Vector(obj.bound_box[i])
        zs.append((obj.matrix_world @ c).z)
    return max(zs)


def get_ground_plane():
    obj = bpy.data.objects.get("Plane")
    if obj and obj.type == "MESH":
        return obj
    for o in bpy.data.objects:
        if o.type == "MESH" and (o.name.startswith("Plane") or "plane" in o.name.lower()):
            return o
    return None


# ════════════════════════════════════════════════════════
#  纹理 / 材质
# ════════════════════════════════════════════════════════

def apply_ground_textures(plane):
    if not plane or plane.type != "MESH":
        return None
    ground_types = []
    try:
        ground_types = [
            d for d in os.listdir(GROUND_TEXTURE_PATH)
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
    if not os.path.isdir(texture_path):
        return None

    diffuse_path = normal_path = roughness_path = displacement_path = None
    try:
        all_files = os.listdir(texture_path)
    except Exception:
        return None

    for file in all_files:
        fl = file.lower()
        if "diff_4k" in fl and fl.endswith(".jpg"):
            diffuse_path = os.path.join(texture_path, file)
        elif "nor_gl_4k" in fl and fl.endswith(".exr"):
            normal_path = os.path.join(texture_path, file)
        elif "rough_4k" in fl and (fl.endswith(".jpg") or fl.endswith(".exr")):
            roughness_path = os.path.join(texture_path, file)
        elif "disp_4k" in fl and (fl.endswith(".jpg") or fl.endswith(".png")):
            displacement_path = os.path.join(texture_path, file)

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
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.location = (-400, 200)
        tex.image = bpy.data.images.load(diffuse_path)
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], principled.inputs["Base Color"])

    if normal_path and os.path.exists(normal_path):
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.location = (-400, 0)
        tex.image = bpy.data.images.load(normal_path)
        nm = nodes.new(type="ShaderNodeNormalMap")
        nm.location = (-200, 0)
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], nm.inputs["Color"])
        links.new(nm.outputs["Normal"], principled.inputs["Normal"])

    if roughness_path and os.path.exists(roughness_path):
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.location = (-400, -200)
        tex.image = bpy.data.images.load(roughness_path)
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], principled.inputs["Roughness"])

    if displacement_path and os.path.exists(displacement_path):
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.location = (-400, -400)
        tex.image = bpy.data.images.load(displacement_path)
        disp = nodes.new(type="ShaderNodeDisplacement")
        disp.location = (-200, -400)
        disp.inputs["Scale"].default_value = 0.05
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], disp.inputs["Height"])
        links.new(disp.outputs["Displacement"], output.inputs["Displacement"])
        if plane.modifiers.get("Subdivision") is None:
            subdiv = plane.modifiers.new(name="Subdivision", type="SUBSURF")
            subdiv.levels = 2
            subdiv.render_levels = 2
        if getattr(mat, "cycles", None) is not None:
            mat.cycles.displacement_method = "BOTH"

    return meta_name


def _jitter_rgb(rgb, amount=0.04):
    return tuple(max(0.05, min(0.92, c + random.uniform(-amount, amount))) for c in rgb)


def _jitter_rgb_dark(rgb, amount=0.014):
    """斜面材质：限制在暗色域，避免抖动成浅黄/浅灰（上限与整体压暗后的调色一致）。"""
    return tuple(max(0.01, min(0.15, c + random.uniform(-amount, amount))) for c in rgb)


def apply_random_color_material(obj, name_prefix="Mat", earth_tone_ramp=False):
    """
    默认：高饱和随机色（用于 Slider 等）。
    earth_tone_ramp=True：暗色大地色系 + 程序化噪声/凹凸（用于 Wedge 斜面，接近水泥/夯土/旧金属等）。
    """
    mat = bpy.data.materials.new(name=f"{name_prefix}_{obj.name}")
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.location = (400, 0)
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (700, 0)
    links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])

    if not earth_tone_ramp:
        r = random.uniform(0.05, 0.95)
        g = random.uniform(0.05, 0.95)
        b = random.uniform(0.05, 0.95)
        principled.inputs["Base Color"].default_value = (r, g, b, 1.0)
        principled.inputs["Roughness"].default_value = random.uniform(0.2, 0.7)
        principled.inputs["Metallic"].default_value = random.uniform(0.0, 0.4)
        return (r, g, b)

    # ─── Wedge：深暗色系 + 多层程序化纹理（Noise + Voronoi 边距）+ 强凹凸/粗糙变化 ───
    # 调色板：暗银灰、深土黄/赭褐、青黑石板等；c_hi 仍明显暗于旧版，避免整体发浅发黄
    palettes = [
        ("dark_charcoal", (0.08, 0.08, 0.09), (0.18, 0.18, 0.20), 0.02, 0.93),
        ("deep_ochre", (0.14, 0.10, 0.06), (0.24, 0.18, 0.11), 0.0, 0.94),
        ("iron_silver", (0.10, 0.10, 0.11), (0.22, 0.22, 0.24), 0.08, 0.88),
        ("blue_black_slate", (0.06, 0.07, 0.09), (0.15, 0.16, 0.19), 0.04, 0.91),
        ("dark_brown_mudstone", (0.11, 0.08, 0.06), (0.20, 0.15, 0.11), 0.0, 0.95),
        ("oxidized_dark_steel", (0.11, 0.10, 0.10), (0.24, 0.23, 0.24), 0.58, 0.74),
    ]
    _name, c_lo, c_hi, m0, rough0 = random.choice(palettes)
    _wedge_darken = random.uniform(0.80, 0.90)
    c_lo = tuple(c * _wedge_darken for c in c_lo)
    c_hi = tuple(c * _wedge_darken for c in c_hi)
    c_lo = _jitter_rgb_dark(c_lo)
    c_hi = _jitter_rgb_dark(c_hi)
    # 保证亮部仍明显深于旧材质（防止 c_hi 被抖动得过亮）
    c_hi = tuple(min(c, 0.26) for c in c_hi)

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = (-1250, 0)
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.location = (-1050, 0)
    su = random.uniform(0.9, 2.8)
    mapping.inputs["Scale"].default_value = (su, su, su)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, random.uniform(0.0, 6.28318))
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    # 大尺度噪声：面状明暗变化
    noise_macro = nodes.new(type="ShaderNodeTexNoise")
    noise_macro.location = (-820, 260)
    noise_macro.inputs["Scale"].default_value = random.uniform(2.2, 9.0)
    noise_macro.inputs["Detail"].default_value = random.uniform(10.0, 15.0)
    noise_macro.inputs["Roughness"].default_value = random.uniform(0.52, 0.68)
    links.new(mapping.outputs["Vector"], noise_macro.inputs["Vector"])

    # 细颗粒噪声：表面砂感
    noise_grain = nodes.new(type="ShaderNodeTexNoise")
    noise_grain.location = (-820, 40)
    noise_grain.inputs["Scale"].default_value = random.uniform(28.0, 72.0)
    noise_grain.inputs["Detail"].default_value = random.uniform(12.0, 15.0)
    noise_grain.inputs["Roughness"].default_value = random.uniform(0.42, 0.58)
    links.new(mapping.outputs["Vector"], noise_grain.inputs["Vector"])

    # Voronoi「到胞元边」距离：石板缝/夯土裂纹感（Blender 4.x: ShaderNodeTexVoronoi）
    vor = nodes.new(type="ShaderNodeTexVoronoi")
    vor.location = (-820, -200)
    vor.voronoi_dimensions = "3D"
    vor.feature = "DISTANCE_TO_EDGE"
    vor.distance = "EUCLIDEAN"
    vor.normalize = True
    vor.inputs["Scale"].default_value = random.uniform(10.0, 32.0)
    links.new(mapping.outputs["Vector"], vor.inputs["Vector"])

    # 归一化各通道到 [0,1] 再混合，便于控制对比度
    mr_n = nodes.new(type="ShaderNodeMapRange")
    mr_n.location = (-560, 200)
    mr_n.inputs["From Min"].default_value = 0.0
    mr_n.inputs["From Max"].default_value = 1.0
    mr_n.inputs["To Min"].default_value = 0.0
    mr_n.inputs["To Max"].default_value = 1.0
    mr_n.clamp = True
    links.new(noise_macro.outputs["Fac"], mr_n.inputs["Value"])

    mr_g = nodes.new(type="ShaderNodeMapRange")
    mr_g.location = (-560, 40)
    mr_g.inputs["From Min"].default_value = 0.0
    mr_g.inputs["From Max"].default_value = 1.0
    mr_g.inputs["To Min"].default_value = 0.0
    mr_g.inputs["To Max"].default_value = 1.0
    mr_g.clamp = True
    links.new(noise_grain.outputs["Fac"], mr_g.inputs["Value"])

    mr_v = nodes.new(type="ShaderNodeMapRange")
    mr_v.location = (-560, -200)
    mr_v.inputs["From Min"].default_value = 0.0
    mr_v.inputs["From Max"].default_value = 1.0
    mr_v.inputs["To Min"].default_value = 0.0
    mr_v.inputs["To Max"].default_value = 1.0
    mr_v.clamp = True
    links.new(vor.outputs["Distance"], mr_v.inputs["Value"])

    mix_ab = nodes.new(type="ShaderNodeMix")
    mix_ab.data_type = "FLOAT"
    mix_ab.blend_type = "MIX"
    mix_ab.location = (-360, 120)
    mix_ab.inputs["Factor"].default_value = random.uniform(0.38, 0.55)
    links.new(mr_n.outputs["Result"], mix_ab.inputs["A"])
    links.new(mr_g.outputs["Result"], mix_ab.inputs["B"])

    mix_abc = nodes.new(type="ShaderNodeMix")
    mix_abc.data_type = "FLOAT"
    mix_abc.blend_type = "MIX"
    mix_abc.location = (-200, 20)
    mix_abc.inputs["Factor"].default_value = random.uniform(0.42, 0.58)
    links.new(mix_ab.outputs["Result"], mix_abc.inputs["A"])
    links.new(mr_v.outputs["Result"], mix_abc.inputs["B"])

    # 拉高中间调对比，让纹理更「显」
    contrast = nodes.new(type="ShaderNodeMapRange")
    contrast.location = (-40, 20)
    contrast.clamp = True
    contrast.inputs["From Min"].default_value = random.uniform(0.22, 0.32)
    contrast.inputs["From Max"].default_value = random.uniform(0.68, 0.82)
    contrast.inputs["To Min"].default_value = 0.0
    contrast.inputs["To Max"].default_value = 1.0
    links.new(mix_abc.outputs["Result"], contrast.inputs["Value"])

    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.location = (160, 20)
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = c_lo + (1.0,)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = c_hi + (1.0,)
    el = ramp.color_ramp.elements.new(random.uniform(0.35, 0.55))
    el.color = (
        (c_lo[0] + c_hi[0]) * 0.5,
        (c_lo[1] + c_hi[1]) * 0.5,
        (c_lo[2] + c_hi[2]) * 0.5,
        1.0,
    )
    links.new(contrast.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])

    # 凹凸：大尺度 + 高频 叠加，强度明显高于旧版
    noise_bump_lo = nodes.new(type="ShaderNodeTexNoise")
    noise_bump_lo.location = (-820, -420)
    noise_bump_lo.inputs["Scale"].default_value = random.uniform(6.0, 18.0)
    noise_bump_lo.inputs["Detail"].default_value = random.uniform(8.0, 12.0)
    noise_bump_lo.inputs["Roughness"].default_value = 0.55
    links.new(mapping.outputs["Vector"], noise_bump_lo.inputs["Vector"])

    noise_bump_hi = nodes.new(type="ShaderNodeTexNoise")
    noise_bump_hi.location = (-820, -580)
    noise_bump_hi.inputs["Scale"].default_value = random.uniform(55.0, 130.0)
    noise_bump_hi.inputs["Detail"].default_value = random.uniform(5.0, 10.0)
    noise_bump_hi.inputs["Roughness"].default_value = 0.48
    links.new(mapping.outputs["Vector"], noise_bump_hi.inputs["Vector"])

    mul_hi = nodes.new(type="ShaderNodeMath")
    mul_hi.location = (-560, -560)
    mul_hi.operation = "MULTIPLY"
    mul_hi.inputs[1].default_value = random.uniform(0.55, 0.85)
    links.new(noise_bump_hi.outputs["Fac"], mul_hi.inputs[0])

    add_bump = nodes.new(type="ShaderNodeMath")
    add_bump.location = (-380, -480)
    add_bump.operation = "ADD"
    links.new(noise_bump_lo.outputs["Fac"], add_bump.inputs[0])
    links.new(mul_hi.outputs["Value"], add_bump.inputs[1])

    bump_norm = nodes.new(type="ShaderNodeMapRange")
    bump_norm.location = (-200, -480)
    bump_norm.clamp = True
    bump_norm.inputs["From Min"].default_value = 0.0
    bump_norm.inputs["From Max"].default_value = 2.0
    bump_norm.inputs["To Min"].default_value = 0.0
    bump_norm.inputs["To Max"].default_value = 1.0
    links.new(add_bump.outputs["Value"], bump_norm.inputs["Value"])

    bump = nodes.new(type="ShaderNodeBump")
    bump.location = (40, -420)
    bump.inputs["Strength"].default_value = random.uniform(0.45, 0.92)
    bump.inputs["Distance"].default_value = random.uniform(0.18, 0.42)
    links.new(bump_norm.outputs["Result"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])

    # 粗糙度随微观噪声变化，高光里也能看出纹理
    rough_mix = nodes.new(type="ShaderNodeMapRange")
    rough_mix.location = (160, -200)
    rough_mix.clamp = True
    rough_mix.inputs["From Min"].default_value = 0.0
    rough_mix.inputs["From Max"].default_value = 1.0
    r_lo = min(1.0, max(0.55, rough0 - 0.12))
    r_hi = min(1.0, rough0 + 0.06)
    rough_mix.inputs["To Min"].default_value = r_lo
    rough_mix.inputs["To Max"].default_value = r_hi
    links.new(noise_grain.outputs["Fac"], rough_mix.inputs["Value"])
    links.new(rough_mix.outputs["Result"], principled.inputs["Roughness"])

    principled.inputs["Metallic"].default_value = min(1.0, max(0.0, m0 + random.uniform(-0.05, 0.05)))

    r = (c_lo[0] + c_hi[0]) * 0.5
    g = (c_lo[1] + c_hi[1]) * 0.5
    b = (c_lo[2] + c_hi[2]) * 0.5
    return (r, g, b)


def apply_ball_textures(obj, texture_folder=None):
    if not obj:
        return None
    if texture_folder is None:
        try:
            subfolders = [f for f in os.listdir(TEXTURE_PATH)
                          if os.path.isdir(os.path.join(TEXTURE_PATH, f))]
        except Exception:
            subfolders = []
        if not subfolders:
            apply_random_color_material(obj, "Roller")
            return None
        texture_folder = random.choice(subfolders)

    pattern_path = os.path.join(TEXTURE_PATH, texture_folder)

    def find_texture(name_base):
        matches = glob.glob(os.path.join(pattern_path, f"{name_base}.*"))
        return matches[0] if matches else None

    diffuse_path = find_texture("pattern")
    normal_path = find_texture("normal")
    roughness_path = find_texture("rough")

    mat = bpy.data.materials.new(name=f"Ball_Mat_{obj.name}_{texture_folder}")
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.location = (0, 0)
    principled.inputs["Roughness"].default_value = 0.22
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (300, 0)
    links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])

    if diffuse_path and os.path.exists(diffuse_path):
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.location = (-300, 100)
        tex.image = bpy.data.images.load(diffuse_path)
        links.new(tex.outputs["Color"], principled.inputs["Base Color"])

    if normal_path and os.path.exists(normal_path):
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.location = (-300, -100)
        tex.image = bpy.data.images.load(normal_path)
        nm = nodes.new(type="ShaderNodeNormalMap")
        nm.location = (-100, -100)
        nm.inputs["Strength"].default_value = random.uniform(0.35, 0.70)
        links.new(tex.outputs["Color"], nm.inputs["Color"])
        links.new(nm.outputs["Normal"], principled.inputs["Normal"])

    if roughness_path and os.path.exists(roughness_path):
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.location = (-300, -300)
        tex.image = bpy.data.images.load(roughness_path)
        bw = nodes.new(type="ShaderNodeRGBToBW")
        bw.location = (-130, -300)
        mul = nodes.new(type="ShaderNodeMath")
        mul.location = (60, -300)
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = random.uniform(0.40, 0.75)
        links.new(tex.outputs["Color"], bw.inputs["Color"])
        links.new(bw.outputs["Val"], mul.inputs[0])
        links.new(mul.outputs["Value"], principled.inputs["Roughness"])

    return texture_folder


def smooth_ball_surface(obj):
    if not obj or obj.type != "MESH":
        return
    for poly in obj.data.polygons:
        poly.use_smooth = True
    subdiv = obj.modifiers.get("BallSmoothSubdiv")
    if subdiv is None:
        subdiv = obj.modifiers.new(name="BallSmoothSubdiv", type="SUBSURF")
    subdiv.levels = 1
    subdiv.render_levels = 2
    subdiv.subdivision_type = "CATMULL_CLARK"


# ════════════════════════════════════════════════════════
#  物理属性
# ════════════════════════════════════════════════════════

def randomize_plane_rigid_body(plane):
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
    return restitution, friction


# ════════════════════════════════════════════════════════
#  斜面 (Wedge) 生成
# ════════════════════════════════════════════════════════

def _wedge_footprint_corners(loc_x, loc_y, base, width, rot_z):
    """
    计算斜面底部矩形在 XY 平面上的四个角点（世界坐标），用于碰撞检测。
    Origin 在底面中心，scale=(base, width, height) 后
    底面从 (-base/2, -width/2) 到 (base/2, width/2)。

    注意本模板中：
      base → X 轴（画面水平方向的展幅）
      width → Y 轴（斜面前后深度方向）
    """
    cos_r = math.cos(rot_z)
    sin_r = math.sin(rot_z)
    local_corners = [
        (-base / 2, -width / 2),
        ( base / 2, -width / 2),
        ( base / 2,  width / 2),
        (-base / 2,  width / 2),
    ]
    world = []
    for lx, ly in local_corners:
        wx = loc_x + lx * cos_r - ly * sin_r
        wy = loc_y + lx * sin_r + ly * cos_r
        world.append((wx, wy))
    return world


def _aabb_from_corners(corners):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), max(xs), min(ys), max(ys)


def _aabb_overlap(a, b):
    ax_min, ax_max, ay_min, ay_max = a
    bx_min, bx_max, by_min, by_max = b
    if ax_max <= bx_min or bx_max <= ax_min:
        return False
    if ay_max <= by_min or by_max <= ay_min:
        return False
    return True


def _within_camera_bounds(aabb, margin=0.5):
    ax_min, ax_max, _, _ = aabb
    return ax_min >= (CAM_X_MIN + margin) and ax_max <= (CAM_X_MAX - margin)


def _hide_template(obj):
    """同时隐藏视口和渲染，确保模板在最终输出中完全不可见。"""
    obj.hide_set(True)
    obj.hide_render = True


def _show_template(obj):
    obj.hide_set(False)
    obj.hide_render = False


def spawn_wedges(ground_plane):
    """
    在底面上放置两个参数各不相同的 Wedge_Template 副本。

    Wedge_Template 几何约定（origin 在底面中心 (0,0,0)）：
      - 底面矩形 4 顶点：(±0.5, ±0.5, 0)
      - 顶棱 2 顶点：(±0.5, 0.5, 1) — Y=+0.5 侧（远离镜头）
      - scale = (base, width, height) 后 →
          底面 (-base/2, -width/2, 0) ~ (base/2, width/2, 0)
          顶棱 (±base/2, width/2, height)
      - 斜面从 (x, width/2, height) 到 (x, -width/2, 0)
        即沿 -Y 方向（朝镜头）倾斜下滑
      - 竖直面（背面）：y = width/2 处

    坐标轴含义：
      base → X 轴（画面水平方向的展幅）
      width → Y 轴（斜面前后深度，即底边→顶棱的水平投影距离）
      height → Z 轴（斜面的竖直高度）
      斜面倾角 θ = arctan(height / width)

    返回两个斜面物体及其元数据。
    """
    template = bpy.data.objects.get("Wedge_Template")
    if not template:
        raise RuntimeError("Wedge_Template not found in blend file!")

    ground_z = _world_top_z_mesh(ground_plane) if ground_plane else 0.0

    wedges = []
    aabbs = []

    side_signs = [-1, 1]
    random.shuffle(side_signs)

    for i in range(2):
        height = random.uniform(*WEDGE_HEIGHT_RANGE)
        base = random.uniform(*WEDGE_BASE_RANGE)
        width = random.uniform(*WEDGE_WIDTH_RANGE)

        if side_signs[i] == -1:
            rot_z = random.uniform(-1.0 * math.pi / 3, math.pi / 2)
        else:
            rot_z = random.uniform(-1.0 * math.pi / 2, math.pi / 3)

        sign = side_signs[i]
        loc_x = sign * random.uniform(*LOCX_RANGE)
        loc_y = random.uniform(*LOCY_RANGE)

        corners = _wedge_footprint_corners(loc_x, loc_y, base, width, rot_z)
        aabb = _aabb_from_corners(corners)

        top_z = ground_z + height
        attempts = 0
        while (not _within_camera_bounds(aabb, margin=0.5)
               or top_z > CAM_Z_MAX - 0.5
               or (aabbs and _aabb_overlap(aabb, aabbs[0]))):
            loc_x = sign * random.uniform(*LOCX_RANGE)
            loc_y = random.uniform(*LOCY_RANGE)
            height = random.uniform(*WEDGE_HEIGHT_RANGE)
            base = random.uniform(*WEDGE_BASE_RANGE)
            top_z = ground_z + height
            corners = _wedge_footprint_corners(loc_x, loc_y, base, width, rot_z)
            aabb = _aabb_from_corners(corners)
            attempts += 1
            if attempts > 200:
                print(f"Warning: wedge {i} placement exhausted, using last position")
                break

        aabbs.append(aabb)

        bpy.ops.object.select_all(action="DESELECT")
        _show_template(template)
        template.select_set(True)
        bpy.context.view_layer.objects.active = template
        bpy.ops.object.duplicate(linked=False)
        wedge = bpy.context.active_object
        wedge.name = f"Wedge_{i}"
        wedge.hide_render = False
        wedge.hide_set(False)
        _hide_template(template)
        template.select_set(False)

        wedge.scale = (base, width, height)
        wedge.location = (loc_x, loc_y, ground_z)
        wedge.rotation_euler = (0, 0, rot_z)

        bpy.context.view_layer.update()

        bpy.ops.object.select_all(action="DESELECT")
        wedge.select_set(True)
        bpy.context.view_layer.objects.active = wedge
        if not wedge.rigid_body:
            bpy.ops.rigidbody.object_add(type="PASSIVE")
        wedge.rigid_body.type = "PASSIVE"
        wedge.rigid_body.collision_shape = "MESH"
        w_friction = random.uniform(*WEDGE_FRICTION_RANGE)
        w_restitution = random.uniform(*WEDGE_RESTITUTION_RANGE)
        wedge.rigid_body.friction = w_friction
        wedge.rigid_body.restitution = w_restitution

        apply_random_color_material(wedge, "Wedge", earth_tone_ramp=True)

        angle_deg = math.degrees(math.atan2(height, width))
        wedges.append({
            "object": wedge,
            "params": {
                "height": height,
                "base": base,
                "width": width,
                "rot_z": rot_z,
                "loc_x": loc_x,
                "loc_y": loc_y,
                "angle_deg": angle_deg,
                "friction": w_friction,
                "restitution": w_restitution,
            }
        })

    return wedges


# ════════════════════════════════════════════════════════
#  斜面上放置物体
# ════════════════════════════════════════════════════════

def _compute_slope_contact_point(wedge_obj, height, width, frac_along_slope=0.3):
    """
    通过读取物体真实的 bounding box，动态推导接触点和法线。
    彻底解耦对模版局部坐标系的硬编码依赖。
    """
    # 1. 获取物体的实际 local bounding box 范围
    bbox = wedge_obj.bound_box
    y_max = max(v[1] for v in bbox)
    y_min = min(v[1] for v in bbox)
    z_max = max(v[2] for v in bbox)
    z_min = min(v[2] for v in bbox)

    # 2. 根据实际 bounding box 找到顶部和底部的中心点，并插值
    norm_top = Vector((0.0, y_max, z_max))
    norm_bottom = Vector((0.0, y_min, z_min))
    norm_pos = norm_top + frac_along_slope * (norm_bottom - norm_top)
    surface_pt = wedge_obj.matrix_world @ norm_pos

    # 3. 计算经过世界缩放后的真实物理维度
    # 如果模版本身不是完美的 1x1x1 比例，物理长高需根据 bbox 比例重新换算
    phys_height = height * (z_max - z_min)
    phys_width = width * (y_max - y_min)
    slope_len = math.sqrt(phys_height ** 2 + phys_width ** 2)

    # 4. 真实的物理斜面夹角
    true_slope_angle = math.atan2(phys_height, phys_width)

    # 5. 基于真实物理维度的局部法线，并转换到世界坐标系
    local_normal = Vector((0.0, -phys_height / slope_len, phys_width / slope_len))
    rot_mat = wedge_obj.matrix_world.to_3x3().normalized()
    world_normal = (rot_mat @ local_normal).normalized()

    return surface_pt, world_normal, true_slope_angle

def place_object_on_wedge(wedge_data, obj_type="slider"):
    """
    在斜面上用 Blender 原生几何体创建滑块（Cube）或滚球（UV Sphere），
    精确放置在斜面表面上方。

    放置策略（从第一性原理推导）：

    对于球体（半径 r）：球心到斜面的法向距离恰好 = r，
        所以 center = surface_point + normal * r

    对于立方体（半边长 h）：放在斜面上时，立方体自身也需要绕 X 轴
        旋转 slope_angle 使其底面与斜面平行。此时立方体中心到斜面
        的法向距离 = h（半边长，即底面到中心的距离），
        所以 center = surface_point + normal * h

    关键：立方体必须旋转到与斜面平行，否则角会戳入斜面。
    旋转方式：绕 **X 轴** 旋转 -slope_angle（因为斜面在 YZ 平面内倾斜，
    从竖直方向绕 X 轴"倒"下来 slope_angle 角度）。
    """
    wedge_obj = wedge_data["object"]
    params = wedge_data["params"]
    height = params["height"]
    width = params["width"]
    rot_z = params["rot_z"]

    frac = random.uniform(0.10, 0.30)
    surface_pt, normal, slope_angle = _compute_slope_contact_point(
        wedge_obj, height, width, frac
    )

    eps = 5e-2
    if obj_type == "slider":
        scale_factor = random.uniform(*SLIDER_SCALE_RANGE)
        half_size = 0.5 * scale_factor

        # 法向偏移 = 半边长（底面中心到立方体几何中心的距离）
        final_pos = surface_pt + normal * half_size * (1 + eps)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=final_pos)
        obj = bpy.context.active_object
        obj.name = "Slider_Active"
        obj.scale = (scale_factor, scale_factor, scale_factor)

        # 让立方体底面平行于斜面：先匹配 wedge 的 Z 旋转，再绕局部 X 轴倾斜
        obj.rotation_euler = (-(math.pi / 2 - slope_angle), 0, rot_z)

        bpy.context.view_layer.update()

        bpy.ops.rigidbody.object_add(type="ACTIVE")
        obj.rigid_body.collision_shape = "BOX"
        vol = scale_factor ** 3
        obj.rigid_body.mass = OBJECT_MATERIAL_DENSITY * vol
        s_friction = random.uniform(*SLIDER_FRICTION_RANGE)
        s_restitution = random.uniform(*SLIDER_RESTITUTION_RANGE)
        obj.rigid_body.friction = s_friction
        obj.rigid_body.restitution = s_restitution

        slider_tf = apply_ball_textures(obj)

        return {
            "name": obj.name,
            "type": "slider",
            "scale_factor": scale_factor,
            "initial_pos": list(final_pos),
            "friction": s_friction,
            "restitution": s_restitution,
            "mass": obj.rigid_body.mass,
            "on_wedge": wedge_obj.name,
            "surface_pt": list(surface_pt),
            "normal": list(normal),
        }

    else:
        radius = random.uniform(*ROLLER_RADIUS_RANGE)

        # 球体：法向偏移 = radius，球心刚好贴在斜面上
        final_pos = surface_pt + normal * radius * (1 + eps)

        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=final_pos)
        obj = bpy.context.active_object
        obj.name = "Roller_Active"
        smooth_ball_surface(obj)

        bpy.context.view_layer.update()

        bpy.ops.rigidbody.object_add(type="ACTIVE")
        obj.rigid_body.collision_shape = "SPHERE"
        vol = (4.0 / 3.0) * math.pi * (radius ** 3)
        obj.rigid_body.mass = OBJECT_MATERIAL_DENSITY * vol
        r_friction = random.uniform(*ROLLER_FRICTION_RANGE)
        r_restitution = random.uniform(*ROLLER_RESTITUTION_RANGE)
        obj.rigid_body.friction = r_friction
        obj.rigid_body.restitution = r_restitution

        tf = apply_ball_textures(obj)

        return {
            "name": obj.name,
            "type": "roller",
            "radius": radius,
            "initial_pos": list(final_pos),
            "friction": r_friction,
            "restitution": r_restitution,
            "mass": obj.rigid_body.mass,
            "texture_folder": tf,
            "on_wedge": wedge_obj.name,
            "surface_pt": list(surface_pt),
            "normal": list(normal),
        }


#  渲染 / GPU / 环境
def setup_gpu_cycles():
    scene = bpy.context.scene
    scene.render.fps = FPS
    if scene.rigidbody_world:
        scene.rigidbody_world.time_scale = 1.0

    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.get_devices()

    device_type = "OPTIX" if "OPTIX" in [d.type for d in prefs.devices] else "CUDA"
    prefs.compute_device_type = device_type
    for dev in prefs.devices:
        dev.use = dev.type in {"CUDA", "OPTIX"}

    scene.cycles.device = "GPU"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 7
    scene.render.use_persistent_data = True
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_min_samples = 48
    scene.cycles.adaptive_threshold = 0.02


def randomize_gravity():
    g_value = random.uniform(1.0, 20.0)
    bpy.context.scene.gravity[2] = -g_value
    return g_value


def randomize_hdri(hdri_path_root):
    if not os.path.exists(hdri_path_root):
        print(f"Error: HDRI path {hdri_path_root} not found")
        return "default"

    hdri_files = [f for f in os.listdir(hdri_path_root) if f.endswith((".hdr", ".exr"))]
    if not hdri_files:
        return "default"
    selected_hdri = random.choice(hdri_files)

    world = bpy.context.scene.world
    world.use_nodes = True
    en_node = world.node_tree.nodes.get("Environment Texture")
    if en_node:
        img = bpy.data.images.load(os.path.join(hdri_path_root, selected_hdri))
        en_node.image = img
    return selected_hdri


def setup_supplementary_key_light():
    scene = bpy.context.scene
    cam = scene.camera
    if not cam:
        return 0.0

    name = "Supplementary_Key_Sun"
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 0.0))
    sun = bpy.context.active_object
    sun.name = name
    sun.data.color = (1.0, 0.98, 0.95)
    energy = random.uniform(50.0, 70.0)
    sun.data.energy = energy

    cam_loc = cam.matrix_world.translation
    light_origin = cam_loc + Vector((0.0, 0.0, 2.5))
    target = Vector((0.0, 0.0, 0.0))
    emit_dir = (target - light_origin).normalized()
    sun.rotation_euler = emit_dir.to_track_quat("-Z", "Y").to_euler()
    return energy


# ════════════════════════════════════════════════════════
#  烘焙 & 渲染
# ════════════════════════════════════════════════════════

def bake_and_render(sample_id, render_dir, metadata):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES

    rbw = scene.rigidbody_world
    if rbw and rbw.point_cache:
        rbw.point_cache.frame_start = 1
        rbw.point_cache.frame_end = TOTAL_FRAMES

    hdri_file_name = metadata["hdri_file"].split(".")[0]
    output_dir_name = f"sample_{sample_id}_g_{metadata['gravity_z']:.2f}_{hdri_file_name}"
    output_path = os.path.abspath(os.path.join(render_dir, output_dir_name))
    os.makedirs(output_path, exist_ok=True)

    print(f"Baking physics for sample {sample_id}...")
    bpy.ops.ptcache.free_bake_all()
    bpy.context.view_layer.update()
    bpy.ops.ptcache.bake_all(bake=True)

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


# ════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════

def main():
    import sys
    argv = sys.argv
    try:
        idx = argv.index("--")
        sample_id = argv[idx + 2]
        render_root = argv[idx + 4]
    except Exception:
        sample_id = "test"
        render_root = "./output"

    setup_gpu_cycles()

    # 确保所有模板在渲染中不可见
    for tpl_name in ("Wedge_Template"):
        tpl = bpy.data.objects.get(tpl_name)
        if tpl:
            _hide_template(tpl)

    g = randomize_gravity()
    hdri = randomize_hdri(".cache/HDRIs")
    sun_energy = setup_supplementary_key_light()

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
        print("Warning: No ground plane found; skip ground textures")

    # 放置两个参数各异的斜面
    wedges = spawn_wedges(ground_plane)

    # 随机决定哪个斜面放滑块、哪个放滚球
    assignments = ["slider", "roller"]
    random.shuffle(assignments)

    sliding_objects_meta = []
    for i, (wedge_data, obj_type) in enumerate(zip(wedges, assignments)):
        meta = place_object_on_wedge(wedge_data, obj_type)
        sliding_objects_meta.append(meta)
        print(f"Placed {obj_type} on {wedge_data['object'].name} at {meta['initial_pos']}")

    """
    wedge_meta = []
    for w in wedges:
        p = w["params"]
        wedge_meta.append({
            "name": w["object"].name,
            "height": p["height"],
            "base": p["base"],
            "width": p["width"],
            "angle_deg": p["angle_deg"],
            "rot_z_rad": p["rot_z"],
            "loc_x": p["loc_x"],
            "loc_y": p["loc_y"],
            "friction": p["friction"],
            "restitution": p["restitution"],
        })
    """

    metadata = {
        "sample_id": sample_id,
        "gravity_z": g,
        "hdri_file": hdri,
        "ground_texture": ground_texture_name,
        # "wedges": wedge_meta,
        # "sliding_objects": sliding_objects_meta,
        "fps": FPS,
        "total_frames": TOTAL_FRAMES,
        "frame_ext": FRAME_EXT,
    }

    bake_and_render(sample_id, render_root, metadata)


if __name__ == "__main__":
    main()
