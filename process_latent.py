import torch
import os
import argparse

def unpack_latent(packed_latent_path):
    for root, dir, files in os.walk(packed_latent_path):
        break
    files = [file for file in files if file.endswith('.pt')]
    assert len(files) == 1, f'Expect 1 packed latent file the given directory path, got {len(files)}'
    results_file = torch.load(packed_latent_path + '/' + files[0])
    root = packed_latent_path + '/latents/'
    if not os.path.isdir(root):
        os.mkdir(root)
    with open(packed_latent_path + '/' + 'list.txt', 'w') as f:
        for key in results_file.keys():
            filename = key.split('/')[-1]
            f.writelines(filename + '\n')
            filename = filename.split('.')[0]
            latent = results_file[key]['latent']
            torch.save(latent, f'{root}/{filename}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--packed_latent_path', '-in', required=True, type=str, help='The packed latent file path derived from GAN inversion.')
    args = parser.parse_args()

    unpack_latent(args.packed_latent_path)