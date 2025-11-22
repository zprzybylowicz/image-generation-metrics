from metrics.fid import FID

if __name__ == "__main__":
    real_dir = "data/real"
    gen_dir = "data/generated/sd15"

    fid_metric = FID(device="cuda", batch_size=32)
    score = fid_metric.from_directories(real_dir, gen_dir)

    print(f"FID between real and generated images: {score:.4f}")
