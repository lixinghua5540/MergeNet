from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import shape


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff"
}


def natural_sort_key(path: Path) -> list:
    """
    对文件名进行自然排序。
    例如：1.png, 2.png, 10.png，而不是 1.png, 10.png, 2.png。
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def find_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    """获取输入文件夹中的全部掩膜图像。"""
    iterator: Iterable[Path]

    if recursive:
        iterator = input_dir.rglob("*")
    else:
        iterator = input_dir.glob("*")

    images = [
        path for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(images, key=natural_sort_key)


def read_binary_mask(
    image_path: Path,
    threshold: int,
    pixel_size_m: float
) -> tuple[np.ndarray, Affine, object | None, str]:
    """
    读取二值掩膜。

    返回：
        binary_mask: 耕地区域为 True，背景为 False。
        transform: 用于栅格矢量化的仿射变换。
        crs: 坐标参考系；普通 PNG/JPG 通常为空。
        georef_mode: 地理参考模式说明。
    """
    with rasterio.open(image_path) as src:
        # 对于 RGB 掩膜，读取第一个波段即可。
        # 黑色背景通常为 0，白色耕地区域通常为 255。
        image = src.read(1)

        binary_mask = image > threshold

        has_valid_georef = (
            src.crs is not None
            and src.transform != Affine.identity()
        )

        if has_valid_georef:
            # GeoTIFF 等带坐标信息的影像：使用原始地理参考。
            transform = src.transform
            crs = src.crs
            georef_mode = "source_georeference"
        else:
            # PNG、JPG 或不带坐标信息的 TIFF：
            # 构造局部米制坐标，使面积仍然能够正确计算。
            transform = from_origin(
                west=0.0,
                north=src.height * pixel_size_m,
                xsize=pixel_size_m,
                ysize=pixel_size_m
            )
            crs = None
            georef_mode = "local_metric_coordinate"

    return binary_mask, transform, crs, georef_mode


def calculate_area_m2(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    计算每个地块的面积，单位为平方米。

    情况 1：普通 PNG/JPG，没有 CRS。
            几何坐标已经由 pixel_size_m 转换为局部米制坐标，
            因此直接计算 geometry.area。

    情况 2：GeoTIFF，具有 CRS。
            将多边形投影至适宜的 UTM 坐标系，再计算面积。
            如果 UTM 估计失败，则使用 EPSG:6933 等积投影。
    """
    if gdf.empty:
        return pd.Series(dtype="float64", index=gdf.index)

    if gdf.crs is None:
        return gdf.geometry.area.astype(float)

    try:
        area_crs = gdf.estimate_utm_crs()

        if area_crs is None:
            area_crs = "EPSG:6933"

    except Exception:
        area_crs = "EPSG:6933"

    projected_gdf = gdf.to_crs(area_crs)

    return projected_gdf.geometry.area.astype(float)


def extract_parcels(
    image_path: Path,
    threshold: int,
    pixel_size_m: float,
    connectivity: int,
    min_area_ha: float
) -> tuple[gpd.GeoDataFrame, str]:
    """
    将单张二值掩膜转为地块多边形。

    每一个连通区域对应一个耕地地块。
    """
    binary_mask, transform, crs, georef_mode = read_binary_mask(
        image_path=image_path,
        threshold=threshold,
        pixel_size_m=pixel_size_m
    )

    polygons = []

    for geometry_mapping, value in shapes(
        source=binary_mask.astype(np.uint8),
        mask=binary_mask,
        connectivity=connectivity,
        transform=transform
    ):
        if int(value) != 1:
            continue

        polygon = shape(geometry_mapping)

        if polygon.is_empty:
            continue

        # 极少数情况下可能存在无效几何，通过 buffer(0) 修复。
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if not polygon.is_empty:
            polygons.append(polygon)

    gdf = gpd.GeoDataFrame(
        {"geometry": polygons},
        geometry="geometry",
        crs=crs
    )

    if gdf.empty:
        gdf["area_m2"] = pd.Series(dtype="float64")
        gdf["area_ha"] = pd.Series(dtype="float64")
        return gdf, georef_mode

    gdf["area_m2"] = calculate_area_m2(gdf)
    gdf["area_ha"] = gdf["area_m2"] / 10000.0

    # 可选：过滤面积非常小的噪声斑块。
    if min_area_ha > 0:
        gdf = gdf[gdf["area_ha"] >= min_area_ha].copy()

    gdf = gdf.reset_index(drop=True)

    return gdf, georef_mode


def save_single_shapefile(
    gdf: gpd.GeoDataFrame,
    shp_path: Path
) -> None:
    """
    保存单张图片对应的 SHP。

    注意：
        Shapefile 的属性字段名不宜过长，因此字段名采用简写。
    """
    if gdf.empty:
        return

    shp_path.parent.mkdir(parents=True, exist_ok=True)

    shp_columns = [
        "gid",        # 全局地块编号
        "tile_no",    # 图片顺序
        "pid",        # 当前图片内部的地块编号
        "src_name",   # 来源图片名称
        "area_m2",    # 面积：平方米
        "area_ha",    # 面积：公顷
        "geometry"
    ]

    gdf[shp_columns].to_file(
        shp_path,
        driver="ESRI Shapefile",
        encoding="UTF-8",
        index=False
    )


def beautify_excel(writer: pd.ExcelWriter) -> None:
    """简单调整 Excel 表头、筛选器和列宽。"""
    workbook = writer.book

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for cell in worksheet[1]:
            cell.font = Font(bold=True)

        for column_cells in worksheet.columns:
            column_index = column_cells[0].column
            column_letter = get_column_letter(column_index)

            max_length = 0

            for cell in column_cells:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))

            worksheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 10),
                40
            )

        header_map = {
            cell.value: cell.column
            for cell in worksheet[1]
        }

        for header, column_index in header_map.items():
            if header is None:
                continue

            header_text = str(header).lower()

            if "area" in header_text or "面积" in header_text:
                for row in range(2, worksheet.max_row + 1):
                    worksheet.cell(
                        row=row,
                        column=column_index
                    ).number_format = "0.0000"


def process_folder(
    input_dir: Path,
    output_excel: Path,
    pixel_size_m: float,
    threshold: int,
    connectivity: int,
    min_area_ha: float,
    save_shp: bool,
    shp_dir: Path,
    recursive: bool
) -> None:
    """
    批量处理文件夹中的全部掩膜图像，并生成总 Excel。
    """
    images = find_images(
        input_dir=input_dir,
        recursive=recursive
    )

    if not images:
        raise FileNotFoundError(
            f"在输入文件夹中没有找到支持的图片：{input_dir}"
        )

    output_excel.parent.mkdir(parents=True, exist_ok=True)

    if save_shp:
        shp_dir.mkdir(parents=True, exist_ok=True)

    all_parcel_rows: list[dict] = []
    image_summary_rows: list[dict] = []

    global_id = 1

    print(f"共发现 {len(images)} 张图片。")
    print("-" * 80)

    for image_order, image_path in enumerate(images, start=1):
        relative_image_path = str(image_path.relative_to(input_dir))

        gdf, georef_mode = extract_parcels(
            image_path=image_path,
            threshold=threshold,
            pixel_size_m=pixel_size_m,
            connectivity=connectivity,
            min_area_ha=min_area_ha
        )

        parcel_count = len(gdf)

        if parcel_count > 0:
            gdf["gid"] = np.arange(
                global_id,
                global_id + parcel_count,
                dtype=int
            )

            gdf["tile_no"] = image_order
            gdf["pid"] = np.arange(1, parcel_count + 1, dtype=int)
            gdf["src_name"] = image_path.name

            # 调整字段顺序。
            gdf = gdf[
                [
                    "gid",
                    "tile_no",
                    "pid",
                    "src_name",
                    "area_m2",
                    "area_ha",
                    "geometry"
                ]
            ]

            for row in gdf.itertuples(index=False):
                all_parcel_rows.append(
                    {
                        "global_id": int(row.gid),
                        "image_order": int(row.tile_no),
                        "image_name": row.src_name,
                        "image_relative_path": relative_image_path,
                        "parcel_id_in_image": int(row.pid),
                        "area_m2": float(row.area_m2),
                        "area_ha": float(row.area_ha),
                        "georeference_mode": georef_mode
                    }
                )

            if save_shp:
                shp_name = (
                    f"{image_order:05d}_"
                    f"{image_path.stem}.shp"
                )

                save_single_shapefile(
                    gdf=gdf,
                    shp_path=shp_dir / shp_name
                )

            global_id += parcel_count

        total_area_ha = (
            float(gdf["area_ha"].sum())
            if parcel_count > 0
            else 0.0
        )

        mean_area_ha = (
            float(gdf["area_ha"].mean())
            if parcel_count > 0
            else 0.0
        )

        min_image_area_ha = (
            float(gdf["area_ha"].min())
            if parcel_count > 0
            else 0.0
        )

        max_image_area_ha = (
            float(gdf["area_ha"].max())
            if parcel_count > 0
            else 0.0
        )

        image_summary_rows.append(
            {
                "image_order": image_order,
                "image_name": image_path.name,
                "image_relative_path": relative_image_path,
                "parcel_count": parcel_count,
                "total_area_ha": total_area_ha,
                "mean_area_ha": mean_area_ha,
                "min_area_ha": min_image_area_ha,
                "max_area_ha": max_image_area_ha,
                "georeference_mode": georef_mode
            }
        )

        print(
            f"[{image_order:>4}/{len(images)}] "
            f"{relative_image_path:<40} "
            f"地块数量：{parcel_count:<5} "
            f"总面积：{total_area_ha:.4f} ha"
        )

    all_parcels_df = pd.DataFrame(
        all_parcel_rows,
        columns=[
            "global_id",
            "image_order",
            "image_name",
            "image_relative_path",
            "parcel_id_in_image",
            "area_m2",
            "area_ha",
            "georeference_mode"
        ]
    )

    image_summary_df = pd.DataFrame(
        image_summary_rows,
        columns=[
            "image_order",
            "image_name",
            "image_relative_path",
            "parcel_count",
            "total_area_ha",
            "mean_area_ha",
            "min_area_ha",
            "max_area_ha",
            "georeference_mode"
        ]
    )

    overall_summary_df = pd.DataFrame(
        [
            {
                "image_count": len(images),
                "parcel_count": len(all_parcels_df),
                "total_area_ha": (
                    float(all_parcels_df["area_ha"].sum())
                    if not all_parcels_df.empty
                    else 0.0
                ),
                "mean_parcel_area_ha": (
                    float(all_parcels_df["area_ha"].mean())
                    if not all_parcels_df.empty
                    else 0.0
                ),
                "min_parcel_area_ha": (
                    float(all_parcels_df["area_ha"].min())
                    if not all_parcels_df.empty
                    else 0.0
                ),
                "max_parcel_area_ha": (
                    float(all_parcels_df["area_ha"].max())
                    if not all_parcels_df.empty
                    else 0.0
                )
            }
        ]
    )

    with pd.ExcelWriter(
        output_excel,
        engine="openpyxl"
    ) as writer:
        all_parcels_df.to_excel(
            writer,
            sheet_name="all_parcels",
            index=False
        )

        image_summary_df.to_excel(
            writer,
            sheet_name="image_summary",
            index=False
        )

        overall_summary_df.to_excel(
            writer,
            sheet_name="overall_summary",
            index=False
        )

        beautify_excel(writer)

    print("-" * 80)
    print("处理完成。")
    print(f"图片数量：{len(images)}")
    print(f"地块总数：{len(all_parcels_df)}")

    if not all_parcels_df.empty:
        print(
            "地块总面积："
            f"{all_parcels_df['area_ha'].sum():.4f} ha"
        )

    print(f"Excel 文件：{output_excel}")

    if save_shp:
        print(f"逐图 SHP 文件夹：{shp_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将文件夹中的二值耕地真值图转换为地块多边形，"
            "并汇总生成 Excel 统计表。"
        )
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="二值耕地真值图所在文件夹。"
    )

    parser.add_argument(
        "--output_excel",
        type=Path,
        required=True,
        help="最终输出的 Excel 文件路径。"
    )

    parser.add_argument(
        "--pixel_size",
        type=float,
        required=True,
        help=(
            "空间分辨率，单位为米/像素。"
            "普通 PNG/JPG 必须正确设置；"
            "带 CRS 的 GeoTIFF 会优先使用自身的地理参考信息。"
        )
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="二值化阈值，默认值为 127。像素值大于该阈值时视为耕地。"
    )

    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=4,
        help=(
            "连通区域定义。默认值为 4。"
            "设置为 8 时，角点接触的像素也会视为同一地块。"
        )
    )

    parser.add_argument(
        "--min_area_ha",
        type=float,
        default=0.0,
        help=(
            "最小地块面积阈值，单位为公顷。"
            "默认值为 0，表示不过滤小地块。"
        )
    )

    parser.add_argument(
        "--save_shp",
        action="store_true",
        help="额外保存每张图片对应的 SHP 文件。"
    )

    parser.add_argument(
        "--shp_dir",
        type=Path,
        default=Path("./per_image_shp"),
        help=(
            "逐图 SHP 输出文件夹。"
            "仅在添加 --save_shp 参数时生效。"
        )
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理输入文件夹中的子文件夹。"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_folder(
        input_dir=args.input_dir,
        output_excel=args.output_excel,
        pixel_size_m=args.pixel_size,
        threshold=args.threshold,
        connectivity=args.connectivity,
        min_area_ha=args.min_area_ha,
        save_shp=args.save_shp,
        shp_dir=args.shp_dir,
        recursive=args.recursive
    )