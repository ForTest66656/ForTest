import torch
import numpy as np
from torch.utils.data import dataset

def get_curve2landmark(dataset_name, num_pts):
    if dataset_name == 'COFW':
        dataset_name = '300W'

    if dataset_name == '300W':
        if num_pts == 68:
            curve2landmark = {
                0: torch.arange(0, 9).long().cuda(),
                1: torch.arange(9, 17).long().cuda(),
                2: torch.arange(17, 22).long().cuda(),
                3: torch.arange(22, 27).long().cuda(),
                4: torch.arange(27, 31).long().cuda(),
                5: torch.arange(31, 36).long().cuda(),
                6: torch.arange(36, 42).long().cuda(),
                7: torch.arange(42, 48).long().cuda(),
                8: torch.arange(48, 55).long().cuda(),
                9: torch.arange(55, 60).long().cuda(),
                10: torch.arange(60, 65).long().cuda(),
                11: torch.arange(65, 68).long().cuda()
            }
        elif num_pts == 46:
            curve2landmark = {
                0: torch.arange(0, 5).long().cuda(),
                1: torch.arange(5, 9).long().cuda(),
                2: torch.arange(9, 12).long().cuda(),
                3: torch.arange(12, 15).long().cuda(),
                4: torch.arange(15, 19).long().cuda(),
                5: torch.arange(19, 22).long().cuda(),
                6: torch.arange(22, 28).long().cuda(),
                7: torch.arange(28, 34).long().cuda(),
                8: torch.arange(34, 38).long().cuda(),
                9: torch.arange(38, 41).long().cuda(),
                10: torch.arange(41, 44).long().cuda(),
                11: torch.arange(44, 46).long().cuda()
            }
        elif num_pts == 50:
            curve2landmark = {
                0: torch.arange(0, 5).long().cuda(),
                1: torch.arange(5, 9).long().cuda(),
                2: torch.arange(9, 12).long().cuda(),
                3: torch.arange(12, 15).long().cuda(),
                4: torch.arange(15, 19).long().cuda(),
                5: torch.arange(19, 22).long().cuda(),
                6: torch.arange(22, 28).long().cuda(),
                7: torch.arange(28, 34).long().cuda(),
                8: torch.arange(34, 39).long().cuda(),
                9: torch.arange(39, 44).long().cuda(),
                10: torch.arange(44, 47).long().cuda(),
                11: torch.arange(47, 50).long().cuda()
            }
        elif num_pts == 72:
            curve2landmark = {
                0: torch.arange(0, 9).long().cuda(),
                1: torch.arange(9, 17).long().cuda(),
                2: torch.arange(17, 22).long().cuda(),
                3: torch.arange(22, 27).long().cuda(),
                4: torch.arange(27, 31).long().cuda(),
                5: torch.arange(31, 36).long().cuda(),
                6: torch.arange(36, 42).long().cuda(),
                7: torch.arange(42, 48).long().cuda(),
                8: torch.arange(48, 55).long().cuda(),
                9: torch.arange(55, 62).long().cuda(),
                10: torch.arange(62, 67).long().cuda(),
                11: torch.arange(67, 72).long().cuda()
            }
        elif num_pts == 131:
            curve2landmark = {
                0: torch.arange(0, 17).long().cuda(),
                1: torch.arange(17, 33).long().cuda(),
                2: torch.arange(33, 42).long().cuda(),
                3: torch.arange(42, 51).long().cuda(),
                4: torch.arange(51, 58).long().cuda(),
                5: torch.arange(58, 67).long().cuda(),
                6: torch.arange(67, 79).long().cuda(),
                7: torch.arange(79, 91).long().cuda(),
                8: torch.arange(91, 104).long().cuda(),
                9: torch.arange(104, 115).long().cuda(),
                10: torch.arange(115, 124).long().cuda(),
                11: torch.arange(124, 131).long().cuda()
            }
        else:
            raise NotImplementedError('[{}] {} points are unavailable.'.format(dataset_name, num_pts))
    elif dataset_name == 'WFLW':
        if num_pts == 96 or num_pts == 98:
            curve2landmark = {
                0: torch.arange(0, 17).long().cuda(),
                1: torch.arange(17, 33).long().cuda(),
                2: torch.arange(33, 42).long().cuda(),
                3: torch.arange(42, 51).long().cuda(),
                4: torch.arange(51, 55).long().cuda(),
                5: torch.arange(55, 60).long().cuda(),
                6: torch.arange(60, 68).long().cuda(),
                7: torch.arange(68, 76).long().cuda(),
                8: torch.arange(76, 83).long().cuda(),
                9: torch.arange(83, 88).long().cuda(),
                10: torch.arange(88, 93).long().cuda(),
                11: torch.arange(93, 96).long().cuda()
            }
        elif num_pts == 54:
            curve2landmark = {
                0: torch.arange(0, 9).long().cuda(),
                1: torch.arange(9, 17).long().cuda(),
                2: torch.arange(17, 22).long().cuda(),
                3: torch.arange(22, 27).long().cuda(),
                4: torch.arange(27, 31).long().cuda(),
                5: torch.arange(31, 34).long().cuda(),
                6: torch.arange(34, 38).long().cuda(),
                7: torch.arange(38, 42).long().cuda(),
                8: torch.arange(42, 46).long().cuda(),
                9: torch.arange(46, 49).long().cuda(),
                10: torch.arange(49, 52).long().cuda(),
                11: torch.arange(52, 54).long().cuda()
            }
    elif dataset_name == 'Helen':
        if num_pts == 194:
            curve2landmark = {
                0: torch.arange(0, 21).long().cuda(),
                1: torch.arange(21, 41).long().cuda(),
                2: torch.arange(41, 58).long().cuda(),
                3: torch.arange(58, 72).long().cuda(),
                4: torch.arange(72, 86).long().cuda(),
                5: torch.arange(86, 100).long().cuda(),
                6: torch.arange(100, 114).long().cuda(),
                7: torch.arange(114, 134).long().cuda(),
                8: torch.arange(134, 154).long().cuda(),
                9: torch.arange(154, 174).long().cuda(),
                10: torch.arange(174, 194).long().cuda()
            }
            # uncomment in HELEN to 300W experiment
            # curve2landmark = {
            #     0: torch.arange(0, 21).long().cuda(),
            #     1: torch.arange(21, 41).long().cuda(),
            #     2: torch.cat([torch.arange(41, 45), torch.arange(54, 58)]).long().cuda(),
            #     3: torch.arange(45, 54).long().cuda(),
            #     4: torch.arange(58, 72).long().cuda(),
            #     5: torch.arange(72, 86).long().cuda(),
            #     6: torch.arange(86, 100).long().cuda(),
            #     7: torch.arange(100, 114).long().cuda(),
            #     8: torch.arange(114, 134).long().cuda(),
            #     9: torch.arange(134, 154).long().cuda(),
            #     10: torch.arange(154, 174).long().cuda(),
            #     11: torch.arange(174, 194).long().cuda()
            # }
        elif num_pts == 98:
            curve2landmark = {
                0: torch.arange(0, 11).long().cuda(),
                1: torch.arange(11, 21).long().cuda(),
                2: torch.arange(21, 30).long().cuda(),
                3: torch.arange(30, 37).long().cuda(),
                4: torch.arange(37, 44).long().cuda(),
                5: torch.arange(44, 51).long().cuda(),
                6: torch.arange(51, 58).long().cuda(),
                7: torch.arange(58, 68).long().cuda(),
                8: torch.arange(68, 78).long().cuda(),
                9: torch.arange(78, 88).long().cuda(),
                10: torch.arange(88, 98).long().cuda()
            }
        elif num_pts == 78:
            curve2landmark = {
                0: torch.arange(0, 8).long().cuda(),
                1: torch.arange(8, 15).long().cuda(),
                2: torch.arange(15, 22).long().cuda(),
                3: torch.arange(22, 28).long().cuda(),
                4: torch.arange(28, 34).long().cuda(),
                5: torch.arange(34, 40).long().cuda(),
                6: torch.arange(40, 46).long().cuda(),
                7: torch.arange(46, 54).long().cuda(),
                8: torch.arange(54, 62).long().cuda(),
                9: torch.arange(62, 70).long().cuda(),
                10: torch.arange(70, 78).long().cuda()
            }
    else:
        raise NotImplementedError('[{}] {} points are unavailable.'.format(dataset_name, num_pts))

    return curve2landmark


def get_sigmaV2(dataset_name, num_pts):
    if dataset_name == 'COFW':
        dataset_name = '300W'

    if dataset_name == '300W':
        if num_pts == 68:
            sigmaV2 = torch.cat([
                torch.tensor([36.1889**2]*9),
                torch.tensor([32.2242**2]*8),
                torch.tensor([11.2157**2]*5),
                torch.tensor([11.2157**2]*5),
                torch.tensor([7.2351**2]*4),
                torch.tensor([6.5614**2]*5),
                torch.tensor([6.4939**2]*6),
                torch.tensor([6.4939**2]*6),
                torch.tensor([12.0834**2]*7),
                torch.tensor([7.9841**2]*5),
                torch.tensor([9.3308**2]*5),
                torch.tensor([3.0845**2]*3)]
            ).cuda()
        elif num_pts == 46:
            sigmaV2 = torch.cat([
                torch.tensor([36.1889**2]*5),
                torch.tensor([32.2242**2]*4),
                torch.tensor([11.2157**2]*3),
                torch.tensor([11.2157**2]*3),
                torch.tensor([7.2351**2]*4),
                torch.tensor([6.5614**2]*3),
                torch.tensor([6.4939**2]*6),
                torch.tensor([6.4939**2]*6),
                torch.tensor([12.0834**2]*4),
                torch.tensor([7.9841**2]*3),
                torch.tensor([9.3308**2]*3),
                torch.tensor([3.0845**2]*2)]
            ).cuda()
        elif num_pts == 50:
            sigmaV2 = torch.cat([
                torch.tensor([36.1889**2]*5),
                torch.tensor([32.2242**2]*4),
                torch.tensor([11.2157**2]*3),
                torch.tensor([11.2157**2]*3),
                torch.tensor([7.2351**2]*4),
                torch.tensor([6.5614**2]*3),
                torch.tensor([6.4939**2]*6),
                torch.tensor([6.4939**2]*6),
                torch.tensor([12.0834**2]*5),
                torch.tensor([13.7814**2]*5),
                torch.tensor([9.3308**2]*3),
                torch.tensor([10.1873**2]*3)]
            ).cuda()
        elif num_pts == 72:
            sigmaV2 = torch.cat([
                torch.tensor([36.1889**2]*9),
                torch.tensor([32.2242**2]*8),
                torch.tensor([11.2157**2]*5),
                torch.tensor([11.2157**2]*5),
                torch.tensor([7.2351**2]*4),
                torch.tensor([6.5614**2]*5),
                torch.tensor([6.4939**2]*6),
                torch.tensor([6.4939**2]*6),
                torch.tensor([12.0834**2]*7),
                torch.tensor([13.7814**2]*7),
                torch.tensor([9.3308**2]*5),
                torch.tensor([10.1873**2]*5)]
            ).cuda()
        elif num_pts == 131:
            sigmaV2 = torch.cat([
                torch.tensor([36.1889**2]*17),
                torch.tensor([32.2242**2]*16),
                torch.tensor([11.2157**2]*9),
                torch.tensor([11.2157**2]*9),
                torch.tensor([7.2351**2]*7),
                torch.tensor([6.5614**2]*9),
                torch.tensor([6.4939**2]*12),
                torch.tensor([6.4939**2]*12),
                torch.tensor([12.0834**2]*13),
                torch.tensor([7.9841**2]*11),
                torch.tensor([9.3308**2]*9),
                torch.tensor([3.0845**2]*7)]
            ).cuda()
        else:
            raise NotImplementedError('[{}] {} points are unavailable.'.format(dataset_name, num_pts))
    elif dataset_name == 'WFLW':
        if num_pts == 96 or num_pts == 98:
            sigmaV2 = torch.cat([
                torch.tensor([55.9466**2]*17),
                torch.tensor([52.0476**2]*16),
                torch.tensor([14.7443**2]*9),
                torch.tensor([14.8420**2]*9),
                torch.tensor([11.2228**2]*4),
                torch.tensor([10.1342**2]*5),
                torch.tensor([10.3297**2]*8),
                torch.tensor([10.3474**2]*8),
                torch.tensor([18.0645**2]*7),
                torch.tensor([14.4043**2]*5),
                torch.tensor([15.5064**2]*5),
                torch.tensor([8.6462**2]*3)]
            ).cuda()
        elif num_pts == 54:
            sigmaV2 = torch.cat([
                torch.tensor([55.9466**2]*9),
                torch.tensor([52.0476**2]*8),
                torch.tensor([14.7443**2]*5),
                torch.tensor([14.8420**2]*5),
                torch.tensor([11.2228**2]*4),
                torch.tensor([10.1342**2]*3),
                torch.tensor([10.3297**2]*4),
                torch.tensor([10.3474**2]*4),
                torch.tensor([18.0645**2]*4),
                torch.tensor([14.4043**2]*3),
                torch.tensor([15.5064**2]*3),
                torch.tensor([8.6462**2]*2)]
            ).cuda()
    elif dataset_name == 'Helen':
        if num_pts == 194:
            sigmaV2 = torch.cat([
                torch.tensor([41.9146**2]*21),
                torch.tensor([40.6727**2]*20),
                torch.tensor([17.1873**2]*17),
                torch.tensor([15.9962**2]*14),
                torch.tensor([16.7327**2]*14),
                torch.tensor([12.6246**2]*14),
                torch.tensor([13.0939**2]*14),
                torch.tensor([8.9854**2]*20),
                torch.tensor([8.7150**2]*20),
                torch.tensor([14.3529**2]*20),
                torch.tensor([14.1262**2]*20)]
            ).cuda()
        elif num_pts == 98:
            sigmaV2 = torch.cat([
                torch.tensor([41.9146**2]*11),
                torch.tensor([40.6727**2]*10),
                torch.tensor([17.1873**2]*9),
                torch.tensor([15.9962**2]*7),
                torch.tensor([16.7327**2]*7),
                torch.tensor([12.6246**2]*7),
                torch.tensor([13.0939**2]*7),
                torch.tensor([8.9854**2]*10),
                torch.tensor([8.7150**2]*10),
                torch.tensor([14.3529**2]*10),
                torch.tensor([14.1262**2]*10)]
            ).cuda()
        elif num_pts == 78:
            sigmaV2 = torch.cat([
                torch.tensor([41.9146**2]*8),
                torch.tensor([40.6727**2]*7),
                torch.tensor([17.1873**2]*7),
                torch.tensor([15.9962**2]*6),
                torch.tensor([16.7327**2]*6),
                torch.tensor([12.6246**2]*6),
                torch.tensor([13.0939**2]*6),
                torch.tensor([8.9854**2]*8),
                torch.tensor([8.7150**2]*8),
                torch.tensor([14.3529**2]*8),
                torch.tensor([14.1262**2]*8)]
            ).cuda()
    else:
        raise NotImplementedError('[{}] {} points are unavailable.'.format(dataset_name, num_pts))

    return sigmaV2


def get_index(dataset_name, num_pts):
    if dataset_name == 'COFW':
        dataset_name = '300W'

    if dataset_name == '300W':
        if num_pts == 68:
            index = list(np.arange(0, 68))
        elif num_pts == 131:
            index = list(np.arange(0, 33, 2)) + list(np.arange(33, 42, 2)) + list(np.arange(42, 51, 2)) +\
                    list(np.arange(51, 58, 2)) + list(np.arange(58, 67, 2)) + list(np.arange(67, 79, 2)) +\
                    list(np.arange(79, 91, 2)) + list(np.arange(91, 104, 2)) + list(np.arange(105, 115, 2)) +\
                    list(np.arange(115, 124, 2)) + list(np.arange(126, 131, 2))
        elif num_pts == 46:
            index = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26,
                     27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                     48, 50, 52, 54, 55, 57, 59, 60, 62, 64, 65, 67]
        elif num_pts == 50:
            index = [0, 2, 4, 6, 8, 9, 11, 13, 16, 17, 19, 21, 22, 24, 26, 
                     27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                     48, 50, 51, 52, 54] + [54, 56, 57, 58, 48] + [60, 62, 64] + [64, 66, 60]
        elif num_pts == 72:
            index = list(np.arange(0, 48)) + list(np.arange(48, 55)) + list(np.arange(54, 60)) +\
                    [48] + list(np.arange(60, 65)) + list(np.arange(64, 68)) + [60]
        else:    
            raise NotImplementedError('[{}] {} points are unavailable.'.format(dataset_name, num_pts))
    elif dataset_name == 'WFLW':
        if num_pts == 98:
            index = np.arange(0, 98)
        elif num_pts == 96:
            index = list(np.arange(0, 96))
        elif num_pts == 54:
            index = list(np.arange(0, 17, 2)) + list(np.arange(18, 33, 2)) + [33, 35, 37, 38, 40] +\
                    [42, 44, 46, 48, 50] + list(np.arange(51, 55)) + list(np.arange(55, 60, 2)) +\
                    list(np.arange(60, 68, 2)) + list(np.arange(68, 76, 2)) + list(np.arange(76, 83, 2)) +\
                    list(np.arange(83, 88, 2)) + list(np.arange(88, 93, 2)) + [93, 95]
        else:
            raise NotImplementedError('[{}] {} points are unavailable.'.format(dataset_name, num_pts))
    elif dataset_name == 'Helen':
        if num_pts == 194:
            index = np.arange(0, 194)
        elif num_pts == 98:
            index = list(np.arange(0, 21, 2)) + [21] + list(np.arange(24, 41, 2)) + list(np.arange(41, 58, 2)) +\
                    list(np.arange(58, 72, 2)) + list(np.arange(72, 86, 2)) + list(np.arange(86, 100, 2)) +\
                    list(np.arange(100, 114, 2)) + list(np.arange(114, 134, 2)) + list(np.arange(134, 154, 2)) +\
                    list(np.arange(154, 174, 2)) + list(np.arange(174, 194, 2))
        elif num_pts == 78:
            index = list(np.arange(0, 21, 3)) + [20, 21] + list(np.arange(25, 41, 3)) + [41, 44, 47, 49, 51, 54, 57] +\
                    [58, 61, 64, 65, 68, 71] + [72, 75, 78, 79, 82, 85] + [86, 89, 92, 93, 96, 99] +\
                    [100, 103, 106, 107, 110, 113] + [114, 116, 119, 122, 124, 126, 129, 132] +\
                    [134, 136, 139, 142, 144, 146, 149, 152] + [154, 156, 159, 162, 164, 166, 169, 162] +\
                    [174, 176, 179, 182, 184, 186, 189, 192]
    else:
        raise NotImplementedError('[{}] {} points are unavailable.'.format(dataset_name, num_pts))

    return index


def get_broadcast_index(dataset_name, num_pts, origin_pts=None):
    if dataset_name == 'COFW':
        dataset_name = '300W'
        
    if dataset_name == '300W':
        if num_pts == 68:
            if origin_pts is None:
                index = list(np.arange(0, 68))
            elif origin_pts == 131:
                index = list(np.arange(0, 33, 2)) + list(np.arange(33, 42, 2)) + list(np.arange(42, 51, 2)) +\
                        list(np.arange(51, 58, 2)) + list(np.arange(58, 67, 2)) + list(np.arange(67, 79, 2)) +\
                        list(np.arange(79, 91, 2)) + list(np.arange(91, 104, 2)) + list(np.arange(105, 115, 2)) +\
                        list(np.arange(115, 124, 2)) + list(np.arange(126, 131, 2))
        elif num_pts == 131:
            index = list(np.arange(0, 33, 2)) + list(np.arange(33, 42, 2)) + list(np.arange(42, 51, 2)) +\
                    list(np.arange(51, 58, 2)) + list(np.arange(58, 67, 2)) + list(np.arange(67, 79, 2)) +\
                    list(np.arange(79, 91, 2)) + list(np.arange(91, 104, 2)) + list(np.arange(105, 115, 2)) +\
                    list(np.arange(115, 124, 2)) + list(np.arange(126, 131, 2))
        elif num_pts == 46:
            index = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26,
                     27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                     48, 50, 52, 54, 55, 57, 59, 60, 62, 64, 65, 67]
        elif num_pts == 50:
            index = [0, 2, 4, 6, 8, 9, 11, 13, 16, 17, 19, 21, 22, 24, 26, 
                     27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                     48, 50, 51, 52, 54] + [55, 57, 58, 59, 61] + [62, 64, 66] + [67, 69, 71]
        elif num_pts == 72:
            index = list(np.arange(0, 48)) + list(np.arange(48, 55)) + list(np.arange(54, 60)) +\
                    [48] + list(np.arange(60, 65)) + list(np.arange(64, 68)) + [60]
        else:    
            raise NotImplementedError('[{}] {} points are unavailable.'.format(dataset_name, num_pts))
    elif dataset_name == 'WFLW':
        if num_pts == 98:
            index = np.arange(0, 98)
        elif num_pts == 96:
            index = list(np.arange(0, 96))
        elif num_pts == 54:
            index = list(np.arange(0, 17, 2)) + list(np.arange(18, 33, 2)) + [33, 35, 37, 38, 40] +\
                    [42, 44, 46, 48, 50] + list(np.arange(51, 55)) + list(np.arange(55, 60, 2)) +\
                    list(np.arange(60, 68, 2)) + list(np.arange(68, 76, 2)) + list(np.arange(76, 83, 2)) +\
                    list(np.arange(83, 88, 2)) + list(np.arange(88, 93, 2)) + [93, 95]
        else:
            raise NotImplementedError('[{}] {} points are unavailable.'.format(dataset_name, num_pts))
    elif dataset_name == 'Helen':
        if num_pts == 194:
            index = np.arange(0, 194)
        elif num_pts == 98:
            index = list(np.arange(0, 21, 2)) + [21] + list(np.arange(24, 41, 2)) + list(np.arange(41, 58, 2)) +\
                    list(np.arange(58, 72, 2)) + list(np.arange(72, 86, 2)) + list(np.arange(86, 100, 2)) +\
                    list(np.arange(100, 114, 2)) + list(np.arange(114, 134, 2)) + list(np.arange(134, 154, 2)) +\
                    list(np.arange(154, 174, 2)) + list(np.arange(174, 194, 2))
        elif num_pts == 78:
            index = list(np.arange(0, 21, 3)) + [20, 21] + list(np.arange(25, 41, 3)) + [41, 44, 47, 49, 51, 54, 57] +\
                    [58, 61, 64, 65, 68, 71] + [72, 75, 78, 79, 82, 85] + [86, 89, 92, 93, 96, 99] +\
                    [100, 103, 106, 107, 110, 113] + [114, 116, 119, 122, 124, 126, 129, 132] +\
                    [134, 136, 139, 142, 144, 146, 149, 152] + [154, 156, 159, 162, 164, 166, 169, 162] +\
                    [174, 176, 179, 182, 184, 186, 189, 192]
    else:
        raise NotImplementedError('[{}] {} points are unavailable.'.format(dataset_name, num_pts))

    return index