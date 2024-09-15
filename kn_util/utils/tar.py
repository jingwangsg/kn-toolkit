import multiprocessing as mp
import os.path as osp
import tarfile


from .system import run_cmd


class TarTool:

    def _compress_worker(self, q, dst, index, chunk_size):
        tar = tarfile.open(dst, "w")
        chunk_bytes = 0
        index.value += 1

        while not q.empty():
            file = q.get()
            file_size = osp.getsize(file)
            if chunk_bytes + file_size >= chunk_size:
                tar.close()
                tar = tarfile.open(f"{dst}.{index.value:08d}", "w")
                chunk_bytes = 0
            else:
                tar.add(file)
                chunk_bytes += osp.getsize(file)
        tar.close()

    @classmethod
    def compress(cls, src, dst, chunk_size=1024**3 * 5, num_processes=16):
        """
        Args:
            src: source directory
            dst: destination file
        """

        file_list = run_cmd(f"find {src} -type f").stdout.strip().split("\n")
        print(f"=> Compressing {len(file_list)} files from {src} to {dst}")

        q = mp.Queue()
        for file in file_list:
            q.put(file)

        index = mp.Value("i", -1)

        with mp.Pool(num_processes) as p:
            p.apply_async(cls._compress_worker, args=(q, dst, index, chunk_size))
        