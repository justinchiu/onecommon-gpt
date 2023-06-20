from tqdm import tqdm
import imgkit
import json
from pathlib import Path

from oc.ocdata import get_data
from oc.dot import Dot, single_board_html

from importlib.resources import files
import oc.data
DATA_DIR = Path(files(oc.data)._paths[0])


with (DATA_DIR / "scenarios.json").open("r") as f:
    scenarios = json.load(f)
boards = {
    scenario['uuid']: scenario
    for scenario in scenarios
}


traindata, validdata = get_data()
data = validdata

for x in tqdm(data):
    scenario_id = x["scenario_id"]

    imgpath0 = Path("oc/data/images") / scenario_id / "0.jpeg"
    imgpath1 = Path("oc/data/images") / scenario_id / "1.jpeg"

    imgpath0.parent.mkdir(parents=True, exist_ok=True)
    imgpath1.parent.mkdir(parents=True, exist_ok=True)

    board0 = [Dot(x) for x in boards[scenario_id]["kbs"][0]]
    board1 = [Dot(x) for x in boards[scenario_id]["kbs"][1]]

    options = dict(
        format = ""
    )

    imgkit.from_string(single_board_html(board0), str(imgpath0))
    imgkit.from_string(single_board_html(board1), str(imgpath1))

