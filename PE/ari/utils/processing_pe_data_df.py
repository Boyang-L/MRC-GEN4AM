import csv
import random

INPUT_FILE = '../../data_preprocessing/data/PE_data_df.csv'
AC_TAG = 'adu_spans'


def transfer_df_to_csv():
    all_data = []
    with open(INPUT_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for l_n, line in enumerate(reader):
            if eval(line['adu_spans']) == []:
                continue
            ac_num = len(eval(line['AC_types']))
            doc_id = line['para_id']
            essay_id = line['essay_id']
            para_id = line['para_id']
            text = line['para_text']
            para_types = line['para_types']
            if ac_num == 0:
                ac_text = text
                data_dict = {'doc_id': doc_id,
                              'essay_id': essay_id,
                              'para_id': para_id,
                              'para_types': para_types,
                              'span_pos': '1',
                              'adu_pos': '0',
                              'aty': 'none',
                              'parent_pos': '0',
                              'afu': 'none',
                              'text': ac_text,
                              'rel_pairs': 'none',
                              'root': -1}
                all_data.append(data_dict)
            else:
                ac_count = 0
                span_count = 1

                aty = 'none'
                aty_list = eval(line['AC_types'])
                afu = 'none'
                afu_list = eval(line['AR_types'])
                boundary_list = [0]
                root_idx, others = get_root_node(eval(line['AR_pairs']), ac_num)

                for idx in eval(line[AC_TAG]):
                    for i in idx:
                        if i not in boundary_list:
                            boundary_list.append(i)

                if len(text) not in boundary_list:
                    boundary_list.append(len(text))
                for i in range(len(boundary_list)):
                    if i == len(boundary_list) - 1:
                        break
                    parent_pos = 0
                    if (boundary_list[i], boundary_list[i + 1]) in eval(line[AC_TAG]):
                        ac_text_list = text.split(' ')[boundary_list[i]: boundary_list[i + 1] + 1]

                        ac_text = ' '.join(t for t in ac_text_list)
                        if root_idx == ac_count:
                            root = 2
                            rel_pairs_all = get_sub_graph_pairs(eval(line['AR_pairs']), ac_num, root_idx)
                            rel_pairs = rel_pairs_all[ac_count]
                            aty = aty_list[ac_count]
                            afu = 'none'

                        elif root_idx == 'none':
                            root = 0
                            rel_pairs = []
                            aty = aty_list[ac_count]

                        else:
                            rel_pairs_all = get_sub_graph_pairs(eval(line['AR_pairs']), ac_num, root_idx)

                            rel_pairs = rel_pairs_all[ac_count]
                            root = 0
                            aty = aty_list[ac_count]
                            for i, rel in enumerate(eval(line['AR_pairs'])):
                                if int(rel[1]) == ac_count:
                                    afu = afu_list[i]

                        for parent, child in eval(line['AR_pairs']):
                            if ac_count == child:
                                parent_pos = parent + 1
                        ac_text = ac_text.replace('<AC>', '')
                        ac_text = ac_text.replace('</AC>', '')
                        data_dict = {'doc_id': doc_id,
                                      'essay_id': essay_id,
                                      'para_id': para_id,
                                      'para_types': para_types,
                                      'span_pos': str(span_count),
                                      'adu_pos': str(ac_count + 1),
                                      'aty': aty,
                                      'parent_pos': str(parent_pos),
                                      'afu': afu,
                                      'text': ac_text,
                                      'rel_pairs': rel_pairs,
                                      'root': root}

                        all_data.append(data_dict)
                        ac_count += 1
                        span_count += 1
                    else:
                        if i == 0:
                            ac_text_list = text.split(' ')[boundary_list[i]: boundary_list[i + 1]]
                            ac_text = ' '.join(t for t in ac_text_list)
                        else:
                            ac_text_list = text.split(' ')[boundary_list[i] + 1: boundary_list[i + 1]]
                            ac_text = ' '.join(t for t in ac_text_list)
                        ac_text = ac_text.replace('<AC>', '')
                        ac_text = ac_text.replace('</AC>', '')
                        data_dict = {'doc_id': doc_id,
                                      'essay_id': essay_id,
                                      'para_id': para_id,
                                      'para_types': para_types,
                                      'span_pos': str(span_count),
                                      'adu_pos': '0',
                                      'aty': 'none',
                                      'parent_pos': '0',
                                      'afu': 'none',
                                      'text': ac_text,
                                      'rel_pairs': 'none',
                                      'root': -1}

                        all_data.append(data_dict)
                        span_count += 1
    return split_train_test_dev(all_data)


def get_root_node(rel, ac_num):
    new_rel = []
    for r in rel:
        new_rel.append((r[1], r[0]))
    rel = new_rel

    if len(rel) >= 1:
        level_list = [-1] * ac_num
        rel_pairs = [[]] * ac_num
        level = 1
        while rel:
            rel_list = []
            s_list = []
            e_list = []
            for r in rel:
                s, e = r
                s_list.append(s)
                e_list.append(e)
            for s_i, s in enumerate(s_list):
                if s not in e_list:
                    level_list[s] = level
                    level_list[e_list[s_i]] = level + 1
                    rel_pairs[s_list[s_i]] = rel_pairs[e_list[s_i]] + rel_pairs[e_list[s_i]]
                    rel_pairs[s_list[s_i]].append((s, e_list[s_i]))
                else:
                    rel_list.append(rel[s_i])
            rel = rel_list
            level = level + 1

        max_index_list = []
        max_level = max(level_list)
        for i, l in enumerate(level_list):
            if l == max_level:
                max_index_list.append(i)
        max_index = max(max_index_list)
        return max_index, max_index_list

    else:
        return 'none', 'none'


def get_sub_graph_pairs(rel, ac_num, main_root):

    new_rel = []
    for r in rel:
        new_rel.append((r[1], r[0]))
    rel = new_rel

    level_list = [-1] * ac_num
    rel_pairs = [[]] * ac_num
    level = 1
    sub_graph = {}
    sub_graph['1'] = []
    sub_graph_main = {}
    sub_graph_main['1'] = []
    level_main = 1
    buffer = [main_root]
    l_buffer = 0
    level_list[main_root] = 1
    while l_buffer != len(buffer):
        l_buffer = len(buffer)
        rel_main = []
        next_buffer = []
        for r in rel:
            if r[1] in buffer:
                next_buffer.append(r[0])
                level_list[r[0]] = level_main + 1
                if str(level_main + 1) not in sub_graph_main.keys():
                    sub_graph_main[str(level_main + 1)] = [] + sub_graph_main[str(level_main)]
                    sub_graph_main[str(level_main + 1)].append((r[0], r[1]))
                else:
                    sub_graph_main[str(level_main + 1)].append((r[0], r[1]))
                rel_main.append(r)
        for rm in rel_main:
            rel.remove(rm)

        buffer = buffer + next_buffer
        level_main = level_main + 1
        # print('sub_graph_main', sub_graph_main)
    main_max_level = max(level_list)
    while rel:
        rel_list = []
        s_list = []
        e_list = []
        for r in rel:
            s, e = r
            s_list.append(s)
            e_list.append(e)

        for e_i, e in enumerate(e_list):
            if e not in s_list:
                level_list[e] = level
                level_list[s_list[e_i]] = level + 1
                # print(sub_graph['1'])
                if str(level + 1) not in sub_graph.keys():
                    sub_graph[str(level + 1)] = [] + sub_graph[str(level)]
                    sub_graph[str(level + 1)].append((s_list[e_i], e))
                else:
                    sub_graph[str(level + 1)].append((s_list[e_i], e))
            else:
                rel_list.append(rel[e_i])
        rel = rel_list
        level = level + 1
    sub_graph4ac = [[]] * ac_num
    for i in range(len(sub_graph4ac)):
        # main sub-graph
        if i in buffer:
            sub_graph4ac[i] = sub_graph_main[str(level_list[i])]
        elif level_list[i] != -1:
            sub_graph4ac[i] = sub_graph[str(level_list[i])] + sub_graph_main[str(main_max_level)]
    sub_graph4ac[main_root] = [(main_root, main_root)]
    return sub_graph4ac


def split_train_test_dev(all_data):

    TEST = [1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604,
            1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622,
            1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640,
            1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658,
            1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676,
            1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694,
            1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712,
            1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730,
            1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748,
            1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766,
            1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784,
            1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802,
            1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820,
            1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838,
            1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856,
            1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874,
            1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892,
            1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910,
            1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928,
            1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946,
            1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964,
            1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982,
            1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
            2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
            2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
    TRAIN = []
    DEV = [6, 7, 8, 9, 37, 38, 39, 40, 41, 49, 50, 51, 63, 64, 65, 66, 67, 103, 104, 105, 106, 107, 145, 146, 147, 148, 150, 151, 152, 153, 224, 225, 226, 227, 228, 240, 241, 242, 243, 244, 374, 375, 376, 377, 490, 491, 492, 493, 494, 496, 497, 498, 499, 522, 523, 524, 525, 526, 556, 557, 558, 686, 687, 688, 689, 698, 699, 700, 701, 727, 728, 729, 730, 731, 745, 746, 747, 748, 749, 757, 758, 759, 760, 761, 896, 897, 898, 899, 900, 925, 926, 927, 928, 986, 987, 988, 989, 990, 1027, 1028, 1029, 1030, 1031, 1109, 1110, 1111, 1112, 1166, 1167, 1168, 1169, 1170, 1259, 1260, 1261, 1262, 1263, 1294, 1295, 1296, 1297, 1412, 1413, 1414, 1415, 1432, 1433, 1434, 1435, 1498, 1499, 1500, 1501, 2044, 2045, 2046, 2047, 2061, 2062, 2063]

    data_train = []
    data_test = []
    data_dev = []

    for line in all_data:
        para_id = int(line['para_id'])
        essay_id = int(line['essay_id'])
        if para_id in TEST:
            data_test.append(line)
        elif para_id in DEV:
            data_dev.append(line)
        else:
            data_train.append(line)

    return data_train, data_dev, data_test


data_train, data_dev, data_test = transfer_df_to_csv()



