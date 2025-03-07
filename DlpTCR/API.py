import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from Model_Predict_Feature_Extraction import deal_file
from DLpTCR_server import save_outputfile

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='DLpTCR Prediction Tool')
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to the input CSV file containing TCR sequences'
    )
    parser.add_argument(
        '--job_dir',
        type=str,
        required=True,
        help='Directory name for storing job results'
    )
    return parser

def create_job_directory(base_path: str) -> Path:
    job_dir = Path('./result') / base_path
    job_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Created job directory: {job_dir}')
    return job_dir

def main():

    try:
        parser = setup_parser()
        args = parser.parse_args()

        job_dir = create_job_directory(args.job_dir)

        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f'Input file not found: {input_path}')

        model_select = "B"
        logger.info(f'Processing with model type: {model_select}')

        error_info, TCRA_cdr3, TCRB_cdr3, Epitope = deal_file(
            str(input_path), 
            str(job_dir), 
            model_select
        )

        output_file_path = save_outputfile(
            str(job_dir), 
            model_select, 
            str(input_path),
            TCRA_cdr3,
            TCRB_cdr3,
            Epitope
        )
        
        logger.info(f'Successfully generated output file: {output_file_path}')
        
    except Exception as e:
        logger.error(f'Error during execution: {str(e)}')
        raise

if __name__ == '__main__':
    main()
