import pandas as pd
from pathlib import Path

csv_path = Path(r"E:\Text to image\IFDroid\API Set.csv")   # 改成你的实际路径
out_txt = Path(r"E:\Text to image\IFDroid\sensitive_426_union.txt")

df = pd.read_csv(csv_path)

apis = (
    df["Union"]
    .dropna()
    .astype(str)
    .str.strip()
    .replace("", pd.NA)
    .dropna()
    .drop_duplicates()
    .tolist()
)

with open(out_txt, "w", encoding="utf-8") as f:
    for api in apis:
        f.write(api + "\n")

print(f"提取完成，共 {len(apis)} 个 API")
print(f"输出文件: {out_txt}")