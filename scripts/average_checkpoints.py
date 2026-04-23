import argparse
import torch
from pathlib import Path

def average_checkpoints(ckpt_dir: str, output_path: str):
    ckpt_paths = list(Path(ckpt_dir).glob("*.ckpt"))
    if not ckpt_paths:
        print(f"no checkpoint found in {ckpt_dir}！")
        return
    
    print(f"found {len(ckpt_paths)} checkpoints, starting averaging...")
    
    base_ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    averaged_state_dict = base_ckpt["state_dict"]
    
    original_dtypes = {}
    for k, v in averaged_state_dict.items():
        original_dtypes[k] = v.dtype
        if v.is_floating_point():
            averaged_state_dict[k] = v.to(torch.float32)
            
    for path in ckpt_paths[1:]:
        print(f"   -> add: {path.name}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        for key in averaged_state_dict.keys():
            if averaged_state_dict[key].is_floating_point():
                averaged_state_dict[key] += ckpt["state_dict"][key].to(torch.float32)
            else:
                averaged_state_dict[key] += ckpt["state_dict"][key]
                
    num_ckpts = len(ckpt_paths)
    for key in averaged_state_dict.keys():
        if averaged_state_dict[key].is_floating_point():
            averaged_state_dict[key] = (averaged_state_dict[key] / num_ckpts).to(original_dtypes[key])
        else:
            averaged_state_dict[key] //= num_ckpts
            
    base_ckpt["state_dict"] = averaged_state_dict
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(base_ckpt, output_file)
    print(f"\nAverage completed! {num_ckpts} models averaged.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average multiple PyTorch Lightning checkpoints safely.")
    parser.add_argument(
        "--ckpt_dir", 
        type=str, 
        required=True, 
        help="Include the path to the directory containing the checkpoints to average"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Include the path and name of the output file (e.g., artifacts/seq2seq/averaged_best.pt)"
    )
    
    args = parser.parse_args()
    average_checkpoints(args.ckpt_dir, args.output_path)