def get_dataset_filepath(dataset_root_folder, dataset_name, npoi, leakage_model):
    if dataset_name == "simulate":
        #We generate simulations every time as its relatively fast
        return "sim"
    if leakage_model == "ID":
        dataset_dict = {
            "ASCAD": {
                700: f"{dataset_root_folder}/ASCAD.h5",

            },
            "ascad-variable": {
                1400: f"{dataset_root_folder}/ascad-variable.h5",
                250000: f"{dataset_root_folder}/ASCADr/atmega8515-raw-traces.h5",
            },
            "ascadv2": {
                2000: f"{dataset_root_folder}/ascadv2-extracted.h5",
                15000: f"{dataset_root_folder}/ascadv2-extracted.h5",
                500: f"{dataset_root_folder}/ascadv2-extracted.h5",
                1600: f"{dataset_root_folder}/ascadv2-extracted.h5",
            },
            "eshard": {
                100: f"{dataset_root_folder}/ESHARD/ESHARD_rpoi/eshard_100poi.h5",
                1400: f"{dataset_root_folder}/eshard.h5",
            },
            "aes_hd_mm": {
                3125: f"{dataset_root_folder}/aes_hd_mm_ext.h5",
            },
            "aes_hd": {
                1250: f"{dataset_root_folder}/aes_hd.h5",
            },

            "cswap_pointer": {
                    1000: f"{dataset_root_folder}/ecc_datasets/cswap_pointer.h5.",
            },        
            "cswap_arithmetic": {
                    8000: f"{dataset_root_folder}/ecc_datasets/cswap_arith.h5.",
                    5000: f"{dataset_root_folder}/ecc_datasets/cswap_arith.h5.",
                    1000: f"{dataset_root_folder}/ecc_datasets/cswap_arith.h5.",
            },
                "ascon": {
                    772: f"{dataset_root_folder}/ascon_cw_unprotected.h5",
            },
        }
    else:
        dataset_dict = {
            "ASCAD": {
                700: f"{dataset_root_folder}/ASCAD.h5",

            },
            "ascad-variable": {
                1400: f"{dataset_root_folder}/ascad-variable.h5",
                250000: f"{dataset_root_folder}/ASCADr/atmega8515-raw-traces.h5",
            },
            "ascadv2": {
                2000: f"{dataset_root_folder}/ascadv2-extracted.h5",
                15000: f"{dataset_root_folder}/ascadv2-extracted.h5",
                500: f"{dataset_root_folder}/ascadv2-extracted.h5",
                1600: f"{dataset_root_folder}/ascadv2-extracted.h5",
            },
            "eshard": {
                100: f"{dataset_root_folder}/ESHARD/ESHARD_rpoi/eshard_100poi.h5",
                1400: f"{dataset_root_folder}/eshard.h5",
            },
            "aes_hd_mm": {
                3125: f"{dataset_root_folder}/aes_hd_mm_ext.h5",
            },
            "aes_hd": {
                1250: f"{dataset_root_folder}/aes_hd.h5",
            },

            "cswap_pointer": {
                    1000: f"{dataset_root_folder}/ecc_datasets/cswap_pointer.h5.",
            },        
            "cswap_arithmetic": {
                    8000: f"{dataset_root_folder}/ecc_datasets/cswap_arith.h5.",
                    5000: f"{dataset_root_folder}/ecc_datasets/cswap_arith.h5.",
                    1000: f"{dataset_root_folder}/ecc_datasets/cswap_arith.h5.",
            },
                "ascon": {
                    772: f"{dataset_root_folder}/ascon_cw_unprotected.h5",
            },
        }
    return dataset_dict[dataset_name][npoi]
