import os
import zipfile

import datasets
from datasets import Features, Value
from huggingface_hub import snapshot_download
from tqdm import tqdm


class HestConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        self.patterns = kwargs.pop("patterns", '*')
        super().__init__(**kwargs)


class HestDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        HestConfig(name="custom_config", version="1.0.0", description="hest config")
    ]
    BUILDER_CONFIG_CLASS = HestConfig

    
    def _info(self):
        return datasets.DatasetInfo(
            description="HEST: A Dataset for Spatial Transcriptomics and Histology Image Analysis",
            homepage="",
            license="CC BY-NC-SA 4.0 Deed",
            features=Features({
                'path': Value('string')
            })
        )

    def _split_generators(self, dl_manager):
        # Download files using the huggingface_hub API
        extracted_files = []
        patterns = self.config.patterns
        local_dir = self._cache_dir_root
        
        snapshot_download(repo_id=self.repo_id, allow_patterns=patterns, repo_type="dataset", local_dir=local_dir)

        seg_dir = os.path.join(local_dir, 'cellvit_seg')
        if os.path.exists(seg_dir):
            print('Unzipping cell vit segmentation...')
            for filename in tqdm([s for s in os.listdir(seg_dir) if s.endswith('.zip')]):
                path_zip = os.path.join(seg_dir, filename)
                          
                with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                    zip_ref.extractall(seg_dir)
                    
        wsi_dir = os.path.join(local_dir, 'wsis')
        if os.path.exists(wsi_dir):
            extracted_files = os.listdir(wsi_dir)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": extracted_files},
        )]

    def _generate_examples(self, filepath):
        idx = 0
        for file in filepath:
            yield idx, {
                'path': file
            }
            idx += 1