import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='End-to-End script to automate COLMAP processing and the full SuGaR pipeline from raw images.'
    )

    # Dataset & Path
    parser.add_argument('-s', '--scene_path', type=str, required=True,
                        help='(Required) Path to the scene directory containing an "input" folder with raw images.')
    
    # COLMAP Options
    parser.add_argument('--skip_colmap', action='store_true',
                        help='Skip COLMAP feature extraction and matching if COLMAP processing is already done.')
    parser.add_argument('--camera', type=str, default='OPENCV',
                        help='Camera model for COLMAP (e.g., OPENCV, PINHOLE). Default is OPENCV.')
    parser.add_argument('--colmap_executable', type=str, default='',
                        help='Path to the COLMAP executable if not in system PATH.')
    parser.add_argument('--resize', action='store_true',
                        help='Resize images (images_2, images_4, images_8) during COLMAP conversion.')

    # SuGaR & 3DGS Options
    parser.add_argument('-r', '--regularization_type', type=str, default='dn_consistency',
                        choices=['dn_consistency', 'sdf', 'density'],
                        help='Regularization type for coarse SuGaR. Default is "dn_consistency".')
    parser.add_argument('--low_poly', action='store_true',
                        help='Use low poly mesh config (200k vertices, 6 Gaussians per triangle).')
    parser.add_argument('--high_poly', action='store_true',
                        help='Use high poly mesh config (1M vertices, 1 Gaussian per triangle).')
    parser.add_argument('--refinement_time', type=str, default=None, choices=['short', 'medium', 'long'],
                        help='Refinement time configuration: short (2k iter), medium (7k iter), long (15k iter).')
    parser.add_argument('-t', '--export_obj', type=str, default='True',
                        help='Export textured .obj mesh file. Default is True.')
    parser.add_argument('--export_ply', type=str, default='True',
                        help='Export refined 3D Gaussians .ply file. Default is True.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Index of GPU device to use. Default is 0.')
    parser.add_argument('--white_background', action='store_true',
                        help='Use white background instead of black for rendering.')

    args = parser.parse_args()
    scene_path = Path(args.scene_path).resolve()

    if not scene_path.exists():
        print(f"[ERROR] Specified scene path does not exist: {scene_path}")
        sys.exit(1)

    # -------------------------------------------------------------
    # Step 1: COLMAP Conversion
    # -------------------------------------------------------------
    if not args.skip_colmap:
        input_dir = scene_path / "input"
        if not input_dir.exists():
            print(f"[ERROR] 'input' folder containing raw images was not found in: {scene_path}")
            print(f"Please place your images in: {input_dir}")
            sys.exit(1)

        print("\n==================================================")
        print(" Step 1 / 2: Running COLMAP Feature Extraction & SfM")
        print("==================================================")

        colmap_exec_arg = f'--colmap_executable "{args.colmap_executable}"' if args.colmap_executable else ''
        resize_arg = '--resize' if args.resize else ''

        convert_cmd = (
            f'python gaussian_splatting/convert.py '
            f'-s "{scene_path}" '
            f'--camera {args.camera} '
            f'--gpu_index {args.gpu} '
            f'{colmap_exec_arg} '
            f'{resize_arg}'
        )
        print(f"[CMD] {convert_cmd}")
        exit_code = os.system(convert_cmd)
        if exit_code != 0:
            print(f"[ERROR] COLMAP processing failed with exit code {exit_code}.")
            sys.exit(exit_code)
    else:
        print("\n[INFO] Skipping COLMAP conversion as requested (--skip_colmap).")

    # -------------------------------------------------------------
    # Step 2: Full SuGaR Pipeline (3DGS -> SuGaR Coarse -> Mesh -> SuGaR Refined)
    # -------------------------------------------------------------
    print("\n==================================================")
    print(" Step 2 / 2: Running SuGaR Full Pipeline")
    print("==================================================")

    low_poly_arg = '--low_poly True' if args.low_poly else ''
    high_poly_arg = '--high_poly True' if args.high_poly else ''
    refine_time_arg = f'--refinement_time {args.refinement_time}' if args.refinement_time else ''
    white_bg_arg = '--white_background True' if args.white_background else ''

    sugar_pipeline_cmd = (
        f'python train_full_pipeline.py '
        f'-s "{scene_path}" '
        f'-r {args.regularization_type} '
        f'{low_poly_arg} '
        f'{high_poly_arg} '
        f'{refine_time_arg} '
        f'-t {args.export_obj} '
        f'--export_ply {args.export_ply} '
        f'--gpu {args.gpu} '
        f'{white_bg_arg}'
    )
    print(f"[CMD] {sugar_pipeline_cmd}")
    exit_code = os.system(sugar_pipeline_cmd)
    if exit_code != 0:
        print(f"[ERROR] SuGaR full pipeline failed with exit code {exit_code}.")
        sys.exit(exit_code)

    print("\n==================================================")
    print(" Complete! End-to-end processing finished successfully.")
    print("==================================================")


if __name__ == "__main__":
    main()
