import os
import torch

def merge_checkpoints(weight_paths: list, output_path: str):
    """
    여러 PC에서 생성된 .pth 체크포인트 가중치의 평균을 계산하여 하나로 합칩니다.
    """
    num_models = len(weight_paths)
    if num_models == 0:
        raise ValueError("❌ 합칠 가중치 파일이 없습니다.")

    print(f"🔄 총 {num_models}개 PC의 가중치 병합을 시작합니다...")
    
    # 첫 번째 PC의 가중치 구조 로드 (기준)
    avg_state_dict = torch.load(weight_paths[0], map_location="cpu")

    # 나머지 PC의 가중치 누적 합산
    for path in weight_paths[1:]:
        state_dict = torch.load(path, map_location="cpu")
        for key in avg_state_dict:
            avg_state_dict[key] += state_dict[key]

    # 평균값 계산
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key] / num_models

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(avg_state_dict, output_path)
    print(f"🎉 가중치 평균 병합 완료! -> {output_path}")

if __name__ == "__main__":
    # 3대 PC의 가중치 파일 경로
    pc_weights = [
        "src/merge/best_model_pc1.pth",
        "src/merge/best_model_pc2.pth",
        "src/merge/best_model_pc3.pth"
    ]
    merged_output = "src/merge/merged_model.pth"
    
    merge_checkpoints(pc_weights, merged_output)