import bpy
import mathutils
import json
import os
import random
import math

# ==========================================
# 1. 경로 설정 및 Mixamo Bone 이름 매핑
# ==========================================
# 실행하는 OS 환경에 맞춰 경로를 확인하세요.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "src", "data", "synthetic_images")
ANNOTATION_FILE = os.path.join(PROJECT_ROOT, "src", "data", "synthetic_annotations.json")
FBX_DIR = os.path.join(PROJECT_ROOT, "src", "data", "fbx_sources")

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# Mixamo Armature 표준 Bone 이름
BONE_MAP = {
    "nose": "mixamorig:Head",
    "r_shoulder": "mixamorig:RightShoulder",
    "r_elbow": "mixamorig:RightArm",
    "r_wrist": "mixamorig:RightHand",
    "l_shoulder": "mixamorig:LeftShoulder",
    "l_elbow": "mixamorig:LeftArm",
    "l_wrist": "mixamorig:LeftHand",
    "r_hip": "mixamorig:RightUpLeg",
    "l_hip": "mixamorig:LeftUpLeg"
}

def clean_scene():
    """초기 Default Cube 및 오브젝트 삭제"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def get_bone_camera_coord(armature_obj, bone_name, camera_obj):
    """Bone의 월드 좌표를 OpenCV 표준 카메라 좌표계(+X Right, +Y Down, +Z Depth)로 변환"""
    pose_bone = armature_obj.pose.bones.get(bone_name)
    if not pose_bone:
        return mathutils.Vector((0.0, 0.0, 0.0))
    
    # 1. Bone World Location
    world_matrix = armature_obj.matrix_world @ pose_bone.matrix
    world_loc = world_matrix.to_translation()
    
    # 2. Camera Local Space Transformation
    cam_inv = camera_obj.matrix_world.inverted()
    cam_loc = cam_inv @ world_loc
    
    # 3. Blender Camera local (+X Right, +Y Up, -Z Forward) 
    # -> OpenCV Frame (+X Right, +Y Down, +Z Forward)
    x_cv = cam_loc.x
    y_cv = -cam_loc.y
    z_cv = -cam_loc.z
    
    return mathutils.Vector((x_cv, y_cv, z_cv))

def is_joint_visible(camera_obj, pt_cv):
    """카메라 정면(+Z > 0)에 위치하는지 검사"""
    if pt_cv.z <= 0:
        return 0
    return 1

def setup_scene_environment():
    """카메라 및 조명 생성"""
    # 카메라 추가
    bpy.ops.object.camera_add(location=(0, -3.0, 1.2), rotation=(math.radians(80), 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    
    # 조명(Point Light) 추가
    bpy.ops.object.light_add(type='POINT', location=(1.5, -2.0, 2.5))
    light = bpy.context.object
    light.data.energy = 1000.0
    
    return camera, light

def randomize_environment(camera, light):
    """도메인 랜더마이징 (카메라 위치, 조명 강도/색상)"""
    camera.location.x = random.uniform(-0.6, 0.6)
    camera.location.y = random.uniform(-4.0, -2.2) # 거리 무작위화
    camera.location.z = random.uniform(0.9, 1.5)
    
    light.location.x = random.uniform(-2.0, 2.0)
    light.location.y = random.uniform(-3.0, -1.0)
    light.location.z = random.uniform(1.5, 3.5)
    light.data.energy = random.uniform(500, 2000)
    light.data.color = (random.random(), random.random(), random.random())

def run_pipeline(total_images_target=1000):
    clean_scene()
    camera, light = setup_scene_environment()
    
    # EEVEE 렌더러 설정 (빠른 대량 렌더링용)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types, "RenderEngineEEVEENext") else 'BLENDER_EEVEE'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.image_settings.file_format = 'JPEG'
    
    fbx_files = [f for f in os.listdir(FBX_DIR) if f.endswith('.fbx')]
    if not fbx_files:
        print(f"[Error] {FBX_DIR} 경로에 .fbx 파일이 없습니다!")
        return

    annotations = []
    generated_count = 0

    for fbx_name in fbx_files:
        fbx_path = os.path.join(FBX_DIR, fbx_name)
        seq_id = os.path.splitext(fbx_name)[0]
        
        # FBX 파일 Import
        bpy.ops.import_scene.fbx(filepath=fbx_path)
        
        # Import된 Armature 찾기
        armature = None
        for obj in bpy.context.scene.objects:
            if obj.type == 'ARMATURE':
                armature = obj
                break
                
        if not armature:
            continue
            
        start_frame = scene.frame_start
        end_frame = scene.frame_end
        
        # 프레임 간격(Step)을 두어 모션 프레임 수집
        for frame_idx in range(start_frame, end_frame, 2):
            scene.frame_set(frame_idx)
            randomize_environment(camera, light)
            bpy.context.view_layer.update()
            
            img_filename = f"synthetic_{generated_count:06d}.jpg"
            img_save_path = os.path.join(OUTPUT_IMAGE_DIR, img_filename)
            scene.render.filepath = img_save_path
            
            # 1. 렌더링 및 이미지 저장
            bpy.ops.render.render(write_still=True)
            
            # 2. 관절 좌표 추출
            raw_coords = {}
            vis_list = []
            for k_name, bone_name in BONE_MAP.items():
                pt_cv = get_bone_camera_coord(armature, bone_name, camera)
                raw_coords[k_name] = pt_cv
                vis_list.append(is_joint_visible(camera, pt_cv))
                
            # 3. 중간점 계산 (Neck & Waist)
            r_sh, l_sh = raw_coords["r_shoulder"], raw_coords["l_shoulder"]
            r_hp, l_hp = raw_coords["r_hip"], raw_coords["l_hip"]
            
            neck = (r_sh + l_sh) / 2.0
            waist = (r_hp + l_hp) / 2.0
            
            # 4. 상체 9개 관절 정렬
            ordered_pts = [
                raw_coords["nose"],
                neck,
                raw_coords["r_shoulder"],
                raw_coords["r_elbow"],
                raw_coords["r_wrist"],
                raw_coords["l_shoulder"],
                raw_coords["l_elbow"],
                raw_coords["l_wrist"],
                waist
            ]
            
            # 5. Waist 좌표 차감을 통한 상대 3D 좌표화
            keypoints_3d = []
            for pt in ordered_pts:
                rel = pt - waist
                keypoints_3d.append([round(rel.x, 5), round(rel.y, 5), round(rel.z, 5)])
                
            # Waist Visiblity 포함 (총 9개 관절)
            final_vis = vis_list[:8] + [1]
            
            ann = {
                "image_path": f"synthetic_images/{img_filename}",
                "keypoints_3d": keypoints_3d,
                "visibility": final_vis,
                "source": "synthetic_gt",
                "sequence_id": seq_id,
                "frame_index": frame_idx
            }
            annotations.append(ann)
            
            generated_count += 1
            if generated_count >= total_images_target:
                break
                
        # 다음 FBX를 위해 Armature 삭제
        bpy.ops.object.select_all(action='DESELECT')
        armature.select_set(True)
        bpy.ops.object.delete()
        
        if generated_count >= total_images_target:
            break

    # 기존 synthetic_annotations.json 파일이 있다면 병합 또는 새로 저장
    with open(ANNOTATION_FILE, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    print(f"✅ 총 {generated_count}개의 Synthetic 데이터 생성 및 저장 완료!")

if __name__ == "__main__":
    run_pipeline(total_images_target=1000)