import os
import shutil
import tempfile
import subprocess
import argparse
import fitz
import math

def get_optimal_hardware_params(page_count):
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        free_vram_mb = int(smi_output.strip().split('\n')[0])
    except (subprocess.CalledProcessError, FileNotFoundError):
        free_vram_mb = 8000 

    cpu_cores = os.cpu_count() or 4
    
    vram_workers = free_vram_mb // 4500
    cpu_workers = max(1, cpu_cores - 1)
    workers = max(1, min(vram_workers, cpu_workers))
    
    chunk_size = max(5, min(25, math.ceil(page_count / (workers * 4))))
    
    return workers, chunk_size

def chunk_and_process(pdf_path, final_output_dir, workers_override=None, chunk_override=None):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input file not found: {pdf_path}")

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    final_md_path = os.path.join(final_output_dir, f"{pdf_name}.md")
    
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    
    heuristic_workers, heuristic_chunk = get_optimal_hardware_params(num_pages)
    active_workers = workers_override if workers_override else heuristic_workers
    active_chunk = chunk_override if chunk_override else heuristic_chunk
    
    print(f"Processing '{pdf_name}' ({num_pages} pages) -> {active_workers} workers, {active_chunk} pages/chunk.")
    
    with tempfile.TemporaryDirectory(dir="/tmp") as temp_base:
        chunk_dir = os.path.join(temp_base, "chunks")
        marker_out_dir = os.path.join(temp_base, "out")
        os.makedirs(chunk_dir)
        os.makedirs(marker_out_dir)

        num_chunks = 0
        for i in range(0, num_pages, active_chunk):
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=i, to_page=min(i + active_chunk - 1, num_pages - 1))
            chunk_doc.save(os.path.join(chunk_dir, f"chunk{num_chunks:04d}.pdf"))
            chunk_doc.close()
            num_chunks += 1
        doc.close()

        custom_env = os.environ.copy()
        custom_env["TORCH_DEVICE"] = "cuda"
        custom_env["LAYOUT_BATCH_SIZE"] = "8"
        custom_env["DETECTION_BATCH_SIZE"] = "16"
        custom_env["RECOGNITION_BATCH_SIZE"] = "16"
        custom_env["FORCE_OCR"] = "1"
        custom_env["OUTPUT_FORMAT"] = "markdown"

        subprocess.run([
            "marker", chunk_dir, marker_out_dir,
            "--workers", str(active_workers)
        ], env=custom_env, check=True)

        os.makedirs(final_output_dir, exist_ok=True)
        final_images_dir = os.path.join(final_output_dir, "images")
        
        with open(final_md_path, "w", encoding="utf-8") as outfile:
            for i in range(num_chunks):
                chunk_folder = os.path.join(marker_out_dir, f"chunk{i:04d}")
                
                if os.path.exists(chunk_folder):
                    md_files = [f for f in os.listdir(chunk_folder) if f.endswith(".md")]
                    if md_files:
                        chunk_md_path = os.path.join(chunk_folder, md_files[0])
                        with open(chunk_md_path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read() + "\n\n---\n\n")

                    images_dir = os.path.join(chunk_folder, "images")
                    if os.path.exists(images_dir):
                        os.makedirs(final_images_dir, exist_ok=True)
                        for img in os.listdir(images_dir):
                            shutil.copy(
                                os.path.join(images_dir, img), 
                                os.path.join(final_images_dir, f"{i:04d}_{img}")
                            )
                else:
                    print(f"Warning: Output directory {chunk_folder} not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process large PDFs with Marker via multiprocessing.")
    parser.add_argument("pdf_path", help="Path to the target PDF file.")
    parser.add_argument("output_dir", help="Destination directory for markdown and extracted images.")
    parser.add_argument("--workers", type=int, help="Override dynamic hardware worker calculation.")
    parser.add_argument("--chunk_size", type=int, help="Override dynamic page chunk calculation.")
    
    args = parser.parse_args()
    
    chunk_and_process(args.pdf_path, args.output_dir, args.workers, args.chunk_size)
