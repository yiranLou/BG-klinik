# run_synergy_analysis.py
"""
Main script to run the complete muscle synergy analysis pipeline.
"""

from newPipeline.src.synergy import SynergyExtractor, SynergyAnalyzer
import argparse


def main():
    # Define paths
    data_path = r"C:\temporary_file\BG_klinik\newPipeline\data\processed\emg_processed\emg_norm.sto"
    output_path = r"C:\temporary_file\BG_klinik\newPipeline\data\processed\synergies"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run muscle synergy analysis')
    parser.add_argument('--n_synergies', type=int, default=None,
                        help='Number of synergies to extract (default: auto-determine)')
    parser.add_argument('--determine_optimal', action='store_true', default=True,
                        help='Automatically determine optimal number of synergies')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU acceleration if available')
    parser.add_argument('--analyze_only', action='store_true', default=False,
                        help='Only run analysis on existing results')

    args = parser.parse_args()

    if not args.analyze_only:
        # Step 1: Extract synergies
        print("=" * 50)
        print("Step 1: Extracting Muscle Synergies")
        print("=" * 50)

        extractor = SynergyExtractor(data_path, output_path)
        extractor.run_extraction_pipeline(
            n_synergies=args.n_synergies,
            determine_optimal=args.determine_optimal
        )

        # Get the latest extraction directory
        import os
        subdirs = [d for d in os.listdir(output_path)
                   if os.path.isdir(os.path.join(output_path, d)) and d.startswith('synergy_extraction')]
        if subdirs:
            latest_dir = sorted(subdirs)[-1]
            analysis_path = os.path.join(output_path, latest_dir)
        else:
            analysis_path = output_path
    else:
        # Find existing results
        import os
        subdirs = [d for d in os.listdir(output_path)
                   if os.path.isdir(os.path.join(output_path, d)) and d.startswith('synergy_extraction')]
        if subdirs:
            latest_dir = sorted(subdirs)[-1]
            analysis_path = os.path.join(output_path, latest_dir)
        else:
            analysis_path = output_path

    # Step 2: Analyze results
    print("\n" + "=" * 50)
    print("Step 2: Analyzing Synergy Results")
    print("=" * 50)

    analyzer = SynergyAnalyzer(analysis_path)
    analyzer.run_full_analysis()

    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print(f"Results saved to: {analysis_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()