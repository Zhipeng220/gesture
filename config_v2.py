# config_v2.py
import json


def get_experiments():
    """
    Defines and returns configurations for all experiments, including new versions
    based on the improvement analysis.
    """
    base_config = {
        'use_skeletal_features': True,
    }

    # MODIFIED: Updated base parameters for new models as per Tier 1 suggestions
    v15_model_params = {
        'model_class': 'DHSGNet_V4',
        'seq_len': 128,  # MODIFIED: (Tier 1-A) Increased from 96 to 128
        'num_layers': 8, 'num_heads': 16, 'embed_dim': 512, 'input_dim': 9,
        'projection_head_dims': [512, 256, 128],
        'num_gnn_layers': 3,  # MODIFIED: (Tier 1-B) Increased from 2 to 3
        'gnn_type': 'HGCN',
        'skeletal_patch_size': 4,
    }

    experiments = {
        # --- Original V16 Experiment for baseline comparison ---
        "12_Exp_V16_Optimized": {
            'batch_size': 8,
            'epochs': 200,
            'optimizer_type': 'adamw_custom',
            'learning_rate': 5e-5,
            'warmup_epochs': 10,
            'classifier_lr_mult': 2.0,
            'scheduler_type': 'cosine',
            'loss_function': 'focal',
            'focal_gamma': 1.8,
            'label_smoothing': 0.05,
            'weight_decay': 1e-2,
            'anatomical_loss_weight': 0.2,
            'mixup_alpha': 0.4,
            'early_stopping_patience': 40,
            'model_params': {
                'seq_len': 96,  # Keep original seq_len for direct comparison
                'num_gnn_layers': 2,
                'dropout': 0.35,
                'stochastic_depth_rate': 0.2,
                'use_arcface': False,
            }
        },

        # --- NEW: V17 Experiment applying all Tier 1 improvements ---
        "13_Exp_V17_Tier1_Improved": {
            'batch_size': 8, 'epochs': 200, 'optimizer_type': 'adamw_custom', 'learning_rate': 5e-5,
            'warmup_epochs': 10, 'classifier_lr_mult': 2.0, 'scheduler_type': 'cosine',
            'loss_function': 'focal', 'focal_gamma': 1.8, 'label_smoothing': 0.05, 'weight_decay': 1e-2,
            'anatomical_loss_weight': 0.2, 'mixup_alpha': 0.4, 'early_stopping_patience': 40,
            'model_params': {
                # Inherits seq_len=128 and num_gnn_layers=3 from new base
                'gnn_residual': True,  # NEW: (Tier 1-B) Enable GNN residuals
                'dropout': 0.25,  # MODIFIED: (Tier 1-C) Reduced main dropout
                'attention_dropout': 0.1,  # NEW: (Tier 1-C) Specific attention dropout
                'stochastic_depth_rate': 0.15,  # MODIFIED: (Tier 1-C) Corresponds to path_dropout
                'use_arcface': False,
            }
        },

        # --- NEW: V18 Experiment applying Tier 2 architectural enhancements ---
        "14_Exp_V18_Tier2_Advanced": {
            'batch_size': 8, 'epochs': 200, 'optimizer_type': 'adamw_custom', 'learning_rate': 5e-5,
            'warmup_epochs': 10, 'classifier_lr_mult': 2.0, 'scheduler_type': 'cosine',
            'loss_function': 'focal', 'focal_gamma': 1.8, 'label_smoothing': 0.05, 'weight_decay': 1e-2,
            'anatomical_loss_weight': 0.2, 'mixup_alpha': 0.4, 'early_stopping_patience': 40,
            'model_params': {
                'gnn_residual': True, 'dropout': 0.25, 'attention_dropout': 0.1, 'stochastic_depth_rate': 0.15,
                # Tier 2 additions below
                'use_temporal_conv': True,  # NEW: (Tier 2-D) Enable TCN
                'tcn_kernel_size': 3,
                'dilations': [1, 2, 4],
                'use_cross_modal_attention': True,  # NEW: (Tier 2-E) Enable cross-modal fusion
                'adaptive_depth': True,  # NEW: (Tier 2-F) Enable adaptive depth
                'layer_selection_threshold': 0.1,
                'use_arcface': False,
            }
        },
        #
        # --- NEW: V18 Pre-training experiment using Self-Supervised Learning ---
        "15_Exp_V18_Pretrain": {
            'mode': 'pretrain',  # NEW: Special mode handled by train script
            'batch_size': 16, 'epochs': 200, 'optimizer_type': 'adamw_custom', 'learning_rate': 1e-4,
            'warmup_epochs': 10, 'scheduler_type': 'cosine', 'weight_decay': 1e-2,
            'loss_function': 'infonce',  # Using contrastive loss
            'contrastive_temperature': 0.07,
            'model_params': {
                # Use advanced T2 architecture for pre-training
                'gnn_residual': True, 'dropout': 0.25, 'attention_dropout': 0.1, 'stochastic_depth_rate': 0.15,
                'use_temporal_conv': True, 'tcn_kernel_size': 3, 'dilations': [1, 2, 4],
                'use_cross_modal_attention': True, 'adaptive_depth': True,
                # Keep depth fixed for pre-training stability
            }
        },

        # --- NEW: V18 Fine-tuning with Knowledge Distillation ---
        "16_Exp_V18_Finetune_KD": {
            'batch_size': 8, 'epochs': 200, 'optimizer_type': 'adamw_custom', 'learning_rate': 2e-5,
            'warmup_epochs': 5, 'scheduler_type': 'cosine', 'loss_function': 'focal',
            # Tier 3 additions below
            'use_knowledge_distillation': True,
            'teacher_models': ['results/pretrained_backbone_15_Exp_V18_Pretrain.pth'],  # IMPORTANT: Update with actual path
            'distillation_alpha': 0.3,
            'distillation_temp': 2.0,
            'model_params': {
                # Assumes we are fine-tuning the best (Tier 2) architecture
                'gnn_residual': True, 'dropout': 0.25, 'attention_dropout': 0.1, 'stochastic_depth_rate': 0.15,
                'use_temporal_conv': True, 'tcn_kernel_size': 3, 'dilations': [1, 2, 4],
                'use_cross_modal_attention': True, 'adaptive_depth': True,
                # 'pretrained_backbone_path': 'results/pretrained_backbone_15_Exp_V18_Pretrain.pth' # Optional: Load pre-trained weights
            }
        },

        # --- NEW: V18 Experiment with Curriculum Learning ---
        "17_Exp_V18_Curriculum": {
            'batch_size': 8, 'epochs': 200, 'optimizer_type': 'adamw_custom', 'learning_rate': 5e-5,
            'warmup_epochs': 10, 'scheduler_type': 'cosine',
            'use_curriculum_learning': True,
            'curriculum_stages': {
                # Define gesture classes for each stage (example)
                'stage1': ['G0', 'G1', 'G2', 'G3'],  # Simplest gestures
                'stage2': ['G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6'],  # Add medium gestures
                'stage3': ['G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13']
                # All gestures
            },
            'model_params': {
                'gnn_residual': True, 'dropout': 0.25, 'attention_dropout': 0.1, 'stochastic_depth_rate': 0.15,
                'use_temporal_conv': True, 'use_cross_modal_attention': True, 'adaptive_depth': True,
            }
        },
    }

    for name, exp_config in experiments.items():
        exp_model_params = exp_config.get('model_params', {})
        # Merge base model params with experiment-specific ones
        final_params = {**v15_model_params, **exp_model_params}
        exp_config['model_params'] = final_params

    return experiments, base_config, v15_model_params


def get_file_paths():
    """Dynamically loads all file and directory paths from an external JSON file."""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Error: `config.json` configuration file not found.")
    paths = config.get("file_paths", {})
    paths["log_dir"] = config.get("log_dir", "logs")
    paths["summary_log_path"] = config.get("summary_log_path", "results/summary.txt")
    if not paths.get("train_csv_path") or not paths.get("test_csv_path"): raise ValueError(
        "Error: `config.json` must contain `train_csv_path` and `test_csv_path`.")
    return paths