import seaborn as sns
try:
    from fau_colors import colors as _FAU, colors_dark as _FAU_D, cmaps as _FAU_CMAPS
    print("tech_dark:", _FAU_CMAPS.tech_dark[:4])
    print("tech:", _FAU_CMAPS.tech[:4])
    print("blend:", sns.color_palette(f"blend:{_FAU_D.tech},{_FAU.tech}", 4).as_hex())
except Exception as e:
    print(e)
