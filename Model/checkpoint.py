# checkpoint.py

import os
import glob
import re
import torch


def save_checkpoint(model, optimizer, scheduler, cfg, step, save_dir, epoch=None):
    try:
        
        os.makedirs(save_dir, exist_ok=True)

        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': step,
            'config': cfg._asdict() if hasattr(cfg, '_asdict') else cfg.__dict__
        }
        if epoch is not None:
            checkpoint_name = f"checkpoint-epoch{epoch+1}-step{step}.pt"
        else:
            checkpoint_name = f"checkpoint-step{step}.pt"
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        print(f"Failed to save checkpoint at step {step}: {e}")

def load_checkpoint(save_dir, model, optimizer, scheduler, device):
    try:
        checkpoint_pattern = os.path.join(save_dir, 'checkpoint-epoch*-step*.pt')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            print("No checkpoint found. Starting training from scratch.")
            return 0, 0  # start_epoch, global_step
        
        
        checkpoint_info = []
        pattern = re.compile(r'checkpoint-epoch(\d+)-step(\d+)\.pt')
        for file in checkpoint_files:
            match = pattern.search(os.path.basename(file))
            if match:
                epoch_num = int(match.group(1))
                step_num = int(match.group(2))
                checkpoint_info.append((epoch_num, step_num, file))
        
        if not checkpoint_info:
            print("No valid checkpoint files found. Starting training from scratch.")
            return 0, 0
        
        
        checkpoint_info.sort(key=lambda x: (x[0], x[1]), reverse=True)
        latest_epoch, latest_step, latest_checkpoint = checkpoint_info[0]
        
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        
        global_step = checkpoint.get('step', 0)
        start_epoch = 29
        
        print(f"Resuming training from epoch {start_epoch}, global_step {global_step}")
        
        return start_epoch, global_step
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Starting training from scratch.")
        return 0, 0