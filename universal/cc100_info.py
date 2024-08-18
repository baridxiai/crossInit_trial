import datasets
LANGS = ['af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'bn_rom', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gn', 'gu', 'ha', 'he', 'hi', 'hi_rom', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lg', 'li', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'my_zaw', 'ne', 'nl', 'no', 'ns', 'om', 'or', 'pa', 'pl', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'sa', 'sc', 'sd', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'ta_rom', 'te', 'te_rom', 'th', 'tl', 'tn', 'tr', 'ug', 'uk', 'ur', 'ur_rom', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh-Hans', 'zh-Hant', 'zu']
with open("./cc100_infor.txt") as file:
    for lang in LANGS:
        test = datasets.load_dataset("cc100",lang)
        output_info = lang +"@@" + "datasets.load_dataset('cc100',lang)@@" + str(len(test))
        file.write(output_info)
        file.write("\n")
