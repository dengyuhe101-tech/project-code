# main.py
import argparse
import sys
import os
import src

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Object Detection Data Analysis & Visualization Toolkit")
    
    parser.add_argument('--plot', action='store_true', 
                        help='Run method comparison plotting')
    parser.add_argument('--analyze', action='store_true', 
                        help='Run data analysis: Knee point & Outliers')
    
    args = parser.parse_args()

    # If no arguments provided, print help
    if not args.plot and not args.analyze:
        parser.print_help()
        return
    
    if args.plot:
        src.run_plot()
        
    if args.analyze:
        src.run_analyze()

if __name__ == "__main__":
    main()