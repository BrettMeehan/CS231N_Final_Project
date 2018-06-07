label_dict = {'clear': 0,
              'partly_cloudy': 1,
              'cloudy': 2,
              'haze': 3,
              'water': 4,
              'slash_burn': 5,
              'blow_down': 6,
              'artisinal_mine': 7,
              'blooming': 8,
              'selective_logging': 9,
              'conventional_mine': 10,
              'primary': 11,
              'bare_ground': 12,
              'cultivation': 13,
              'road': 14,
              'agriculture': 15,
              'habitation': 16
              }
def labels_to_int(labels):
    return list(map(label_dict.get, labels))
