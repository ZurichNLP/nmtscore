import logging
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent))
from experiments import paraphrase_tasks

logging.basicConfig(level=logging.INFO)


def download_file(url: str, to_path: Path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(to_path, 'wb') as file, tqdm(
        desc=str(to_path),
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


data_dir = Path(__file__).parent.parent / "data"

logging.info("Downloading English paraphrases data (MRPC, ETPC)")
etpc_dir = data_dir / "etpc"
etpc_dir.mkdir(exist_ok=True)
download_file("https://github.com/venelink/ETPC/raw/master/Corpus/text_pairs.xml", etpc_dir / "text_pairs.xml")
download_file("https://github.com/venelink/ETPC/raw/master/Corpus/textual_np_neg.xml", etpc_dir / "textual_np_neg.xml")
download_file("https://github.com/venelink/ETPC/raw/master/Corpus/textual_np_pos.xml", etpc_dir / "textual_np_pos.xml")
download_file("https://github.com/venelink/ETPC/raw/master/Corpus/textual_paraphrases.xml", etpc_dir / "textual_paraphrases.xml")
# Fix some encoding errors
for xml_path in [
    etpc_dir / "text_pairs.xml",
    etpc_dir / "textual_np_neg.xml",
    etpc_dir / "textual_np_pos.xml",
    etpc_dir / "textual_paraphrases.xml",
]:
    with open(xml_path) as f:
        text = f.read()
    text = text.replace("\x12", "'")
    with open(xml_path, "w") as f:
        f.write(text)
paraphrase_tasks.ETPCTask().get_samples()
paraphrase_tasks.MRPCTask("validation").get_samples()
paraphrase_tasks.MRPCTask("test").get_samples()

logging.info("Downloading Russian paraphrases data")
russian_dir = data_dir / "russian_paraphrases"
russian_dir.mkdir(exist_ok=True)
download_file("http://paraphraser.ru/download/get?file_id=5", russian_dir / "paraphrases_gold.zip")
with zipfile.ZipFile(russian_dir / "paraphrases_gold.zip", 'r') as f:
    f.extractall(russian_dir)
paraphrase_tasks.RussianParaphraseClassificationTask().get_samples()

logging.info("Downloading Finnish paraphrases data")
paraphrase_tasks.FinnishParaphraseClassificationTask().get_samples()

logging.info("Downloading Swedish paraphrases data")
swedish_dir = data_dir / "swedish_paraphrases"
swedish_dir.mkdir(exist_ok=True)
download_file("https://github.com/TurkuNLP/Turku-paraphrase-corpus/raw/main/data-sv/test.json", swedish_dir / "test.json")
paraphrase_tasks.SwedishParaphraseClassificationTask().get_samples()

logging.info("Downloading PAWS-X")
for language in ["en", "de", "es", "fr", "ja", "zh"]:
    paraphrase_tasks.PAWSXParaphraseClassificationTask(language, "validation").get_samples()
    paraphrase_tasks.PAWSXParaphraseClassificationTask(language, "test").get_samples()
