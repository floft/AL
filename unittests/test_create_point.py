from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from sklearn.cluster import KMeans

from al import AL
from config import Config
from features import create_point


class CreatePointTest(TestCase):
    # Testing the create_point method

    def test_create_point_nofilter_noperson_nogpsfeatures(self):
        """Test with no lowpass filtering, no person features, and no gps features."""

        # arrange
        test_al = Mock(spec=AL)

        test_conf = Mock(spec=Config)
        test_conf.filter_data = False
        test_conf.fftfeatures = 0
        test_conf.local = 1
        test_conf.sfeatures = 0
        test_conf.gpsfeatures = 0
        test_conf.samplesize = 10

        test_al.conf = test_conf

        # Set data values:
        test_al.yaw = [0.3640372818564943, -2.322949529941176, 2.008221392962083, 4.907929027158476, 0.9122510136974293, -2.418333798810224, -3.1923166549519175, -4.175033262925778, -4.965486689737542, -1.3906230749752946]
        test_al.pitch = [-3.162968295986767, -4.666647948132239, 4.520716399063069, -3.154039655551234, 0.4182125473423346, -4.168318961881615, -2.5980476073613357, -1.6685651057525641, 3.2371457399291863, 4.194327681646078]
        test_al.roll = [3.9380543732275832, 4.9065900139325755, 1.559575817947608, 3.748158037105849, -1.7157110188361768, 4.875466947054095, -4.4476576053834185, -0.9891119438730014, 1.2696021092024594, 3.5692231813588506]

        test_al.rotx = [9.660077729122644, 1.6355327763219947, -8.94717268162079, -9.964566797618609, 7.897804661527129, -5.590226730125809, 0.3755598247586498, 3.3920453424214223, 3.891498031535436, -8.623797657305122]
        test_al.roty = [-2.0583837392105897, -3.7471621395043764, -9.6355809937429, 0.34834154289583097, 8.825831354570276, -9.995443704445035, 9.281055093155675, 8.263780472795656, 2.7874354315583183, 9.768673315076736]
        test_al.rotz = [2.447353180234547, -6.381022845232042, 8.110399171396288, -8.73097052902153, 0.13941149979300604, 9.928350186964845, 5.5885763896398455, -8.166959369809899, -5.800236031015102, 0.9586844281065137]

        test_al.accx = [-0.5344997627576218, -0.05963941376376036, -0.6846615580872408, 0.3129749488965732, -0.7512729528605864, -0.2289609070868066, 0.08833049044632557, 0.5395261728868226, 0.21064853543393292, 0.13327859486500748]
        test_al.accy = [-0.7787597015645251, 0.7368678712027816, -0.49432635588936247, 0.9604264402547806, -0.6839960928511306, 0.7196290960057685, 0.7853032115413037, 0.6614017016923766, 0.979414274628825, -0.6869694865247487]
        test_al.accz = [-0.24980654859071416, -0.5804666701154173, -0.5046045365473024, -0.279279706155791, 0.5399438139166117, 0.6843085375964921, 0.2215236755012131, -0.29441180878723716, -0.7834837503762135, 0.45188280622940913]
        test_al.acctotal = [0.9770158549827259, 0.9399322711769635, 0.9837406840999424, 1.0480312113900745, 1.15056552482781, 1.0191012253056821, 0.820716850324849, 0.9028947975178757, 1.2717986923765667, 0.832999597737618]

        test_al.latitude = [45.720049024287654, 45.16015407024762, 45.8461484021051, 45.36231411218953, 45.55199912734816, 45.9327322235666, 45.715660777520455, 45.313824413786, 45.67700785441802, 45.287249867092605]
        test_al.longitude = [-115.33785147412178, -115.8627325962175, -115.7079428458205, -115.74277058776053, -115.95217729730174, -115.99332137643903, -115.28416252514502, -115.14186050672829, -115.67576853665538, -115.07208321504989]
        test_al.altitude = [810.2213245992733, 828.020444363237, 840.4794306806942, 824.2303516812959, 819.4818356859511, 829.9659440293694, 818.1985404664574, 837.7380673702816, 806.7219793481587, 816.3683413903411]

        test_al.course = [225.6813948563143, 292.2058167861145, 6.273659084230401, 274.7506368351248, 126.60364340486748, 248.9372759553409, 261.9747954153205, 62.310980695282076, 270.95761178488016, 129.44431108367675]
        test_al.speed = [11.571468342724664, 14.280815369399086, 12.796382917195837, 6.714116530364987, 1.5223602512857481, 9.56722659453084, 6.422785860176224, 5.220575145550258, 10.445374918044397, 2.0756946006433474]

        test_al.hacc = [22.592997936520113, 25.332248443829698, 16.629675669378507, 10.927633865659319, 11.560839234746618, 14.409677626228634, 23.759149321008934, 13.538402787000809, 16.997563118024964, 22.94508413815179]
        test_al.vacc = [14.33151360551745, 13.064804053288855, 20.095012424826972, 26.14868844300875, 11.348279679080182, 21.913967271022265, 13.886057128399077, 17.249532688723097, 28.46768752255568, 12.443856685743517]

        test_al.minlat = 45.16015407024762
        test_al.maxlat = 45.9327322235666
        test_al.minlong = -115.99332137643903
        test_al.maxlong = -115.07208321504989

        test_stamp = datetime(2021, 1, 7, 12, 0, 0)

        test_filename = Mock(spec=str)
        test_person_stats = Mock(spec=np.ndarray)
        test_clusters = [Mock(spec=KMeans) for _ in range(5)]

        expected_output = [
            # statistical features for motion:
            4.907929027158476, -4.965486689737542, -10.27230429566745, -1.027230429566745, -1.8567863024582354, 2.6657181727016415, 2.3706416643757002, 8.398953385482631, 2.8980947854552017, 2.4602720867882923, 2.269535272391136, 4, 3, 0.0, 8.398953385482631, 13.969292770675349, 171.09647552200693, 4.104567668649347, -2.821270381055136, 0.5739005792449817, -0.5745589612322717, 94.54155740910508, 6.4569265150699735, 9.454155740910508, 0.6059475607678763, 2.596238300061732, 1.3229029575347735, 2.4602720867882923, 1.5316705404404074, 1,  # yaw
            4.520716399063069, -4.666647948132239, -7.048185206685087, -0.7048185206685087, -2.13330635655695, 3.1788989942646424, 3.2000570179579766, 11.235436577955, 3.351930276416113, 3.0379352901309407, 2.284177098449977, 5, 5, 0.0, 11.235436577955, 18.198860133165873, 204.12349093550625, 6.400114035915953, -4.755735239813027, 0.48323597509698135, -1.3829885991772943, 117.32205725032344, 8.6686943272358, 11.732205725032344, -0.06801342844239092, 3.8763589344000327, 2.8371408231272883, 3.0379352901309407, 1.416469466996037, 7,  # pitch
            4.9065900139325755, -4.4476576053834185, 16.714189911736423, 1.6714189911736423, 2.5643994996532293, 3.1019151047921616, 3.65869060923235, 8.99525352720595, 2.999208816872535, 2.5360795193621484, 1.8423611604876098, 6, 6, 0.0, 8.99525352720595, -19.268525546111334, 185.66388050442782, 4.927166317100585, 1.79440872259477, -0.7142140193693516, -0.705433721659356, 117.88894971261865, 8.549928710856127, 11.788894971261865, -0.18559639631986186, 3.988798268540031, 2.501234819534921, 2.5360795193621484, 1.6011103018461295, 4,  # roll

            9.660077729122644, -9.964566797618609, -6.273245500983056, -0.6273245500983056, 1.0055463005403222, 5.99782822323576, 6.744015695826469, 46.61878118589529, 6.827794752765734, 6.123293133255421, 6.744015695826469, 5, 5, 0.0, 46.61878118589529, -14.540569818320739, 3427.527997510462, 12.515295688840558, -10.883990992693937, -0.04568146818618438, -1.4229001844724387, 470.12317277051324, 12.849352150573948, 47.012317277051324, -0.180968372226115, 8.108007536449081, 5.660066446680291, 6.123293133255421, 3.0206062951206136, 4,  # rotx
            9.768673315076736, -9.995443704445035, 13.838546633149589, 1.3838546633149589, 1.5678884872270746, 6.47116877869554, 8.544805913682966, 52.75224132578768, 7.263073820758514, 6.401500470116373, 6.976917426455891, 3, 3, 0.0, 52.75224132578768, -116.5952963921965, 4649.059819852062, 12.572993494074652, 5.248436857784012, -0.3043124047165786, -1.3293583623893168, 546.6729505496625, 13.356972463822759, 54.667295054966246, -0.04499837761226175, 8.623471222730808, 6.194804043539673, 6.401500470116373, 3.4311853719797094, 3,  # roty
            9.928350186964845, -8.73097052902153, -1.9064139189435272, -0.19064139189435272, 0.5490479639497599, 5.625196363121362, 6.090629438123573, 41.94367680955961, 6.476393812111769, 5.663324641500233, 6.639677402073332, 6, 6, 0.0, 41.94367680955961, 27.694507484640177, 2822.468831667595, 11.969599234871888, -33.971603688776966, 0.1019515654345067, -1.3956609366405688, 419.80020949863024, 11.178700623243325, 41.980020949863025, -0.23545981205543975, 9.560160201465585, 4.507316086504126, 5.663324641500233, 3.141724178621649, 3,  # rotz

            0.5395261728868226, -0.7512729528605864, -0.9742758520273542, -0.09742758520273542, 0.014345538341282604, 0.35437933370846775, 0.2709679279916899, 0.1738901183012857, 0.41700134088667595, 0.3619369679962628, 0.2709679279916899, 3, 5, 0.0, 0.1738901183012857, -0.019227662538859432, 0.05444809526705274, 0.7451482981915547, -4.280115739489436, -0.26516374061111647, -1.199335155079442, 1.8338225265972197, -11.607064993310107, 0.18338225265972197, -0.07239707410017787, 0.5398681784252821, 0.30039533836423765, 0.3619369679962628, 0.2071032339171891, 4,  # accx
            0.979414274628825, -0.7787597015645251, 2.1989909584960694, 0.21989909584960693, 0.6905153988490725, 0.7487094232155602, 0.7282484836042751, 0.5303711734779502, 0.7282658673025602, 0.704729604045639, 0.2794049585927303, 8, 6, 0.0, 0.5303711734779502, -0.14706482610055271, 0.3506233924567191, 1.4692993043924343, 3.311818379647339, -0.3807491536037002, -1.753532197954545, 5.787267858334247, -2.6583202836609763, 0.5787267858334247, -0.6080752475001567, 1.0470660308267654, 0.6354409714347219, 0.704729604045639, 0.18365009844709348, 5,  # accy
            0.6843085375964921, -0.7834837503762135, -0.7943941873289495, -0.07943941873289495, -0.2645431273732526, 0.4589711853816401, 0.4782436713883558, 0.23824749533666142, 0.4881060287854079, 0.4430833016350612, 0.4009951728083152, 4, 3, 0.0, 0.23824749533666142, 0.028590784886687558, 0.0921289018001904, 0.9564873427767115, -6.144380668577183, 0.24585755734175058, -1.3769226882805197, 2.445581165852816, -7.517193301045982, 0.24455811658528162, 0.3187069788186626, 0.47762157486832674, 0.34163180870876997, 0.4430833016350612, 0.20475517856414482, 1,  # accz
            1.2717986923765667, 0.820716850324849, 9.94679670974011, 0.9946796709740109, 0.9803782695413341, 0.9946796709740109, 0.9803782695413341, 0.017279507229461918, 0.1314515394716316, 0.10215559400081804, 0.07256820693609939, 4, 4, 0.0, 0.017279507229461918, 0.0014800666170776005, 0.0008175674516400893, 0.14513641387219878, 0.13215464566890317, 0.6516047376793929, -0.26182698074777466, 10.066671550784282, -0.11944926288588531, 1.006667155078428, -0.29255821102380936, 0.1630496054055818, 0.13772013834180355, 0.10215559400081804, 0.08272691124296826, 4,  # acctotal

            # 2D features:
            -0.0014112361083133225, -0.32241190889823623, 0.2581391624287407, -2.698649420740387, -0.17007890569968714, -2.376237511842151, 0.6966505372389256, -0.31732668105234235, -0.2248167500814314, -0.017988166469840472, -0.29614427705120416, 0.2993385145825019, 0.3006830799334257, -2.011179213413264, -0.19086412063677605, -0.43668315820395287, -0.38290114982029205, 1.5744960552093117,

            # GPS statistical features:
            45.9327322235666, 45.16015407024762, 455.56713987256177, 45.556713987256174, 45.614503490883095, 45.556713987256174, 45.614503490883095, 0.061825459153484884, 0.24864725848777194, 0.22160566912339164, 0.24191714495778527, 7, 7, 0.0, 0.061825459153484884, -0.0016079471624167695, 0.006347986581188151, 0.4062246105016527, 0.005457971761469174, -0.10459736597507543, -1.3392613260619854, 20754.760147758163, 33.17091838978416, 2075.476014775816, -0.5205399522048085, 0.40799899165868836, 0.14656878549018248, 0.22160566912339164, 0.11276695688834894, 3,  # latitude
            -115.07208321504989, -115.99332137643903, -1155.7706709612396, -115.57706709612395, -115.69185569123795, 115.57706709612395, 115.69185569123795, 0.1037338847322395, 0.3220774514495535, 0.2944621326901796, 0.2808936456324389, 2, 4, 0.0, 0.1037338847322395, 0.009405457347791843, 0.016916053846571667, 0.5785700710724768, -0.002786689951058248, 0.28151332423737574, -1.4279810581881984, 133581.62172426673, 41.257399647154436, 13358.162172426673, 0.13806656369860437, 0.32823373603939704, 0.24703361433556775, 0.2944621326901796, 0.1304834746003897, 2,  # longitude
            840.4794306806942, 806.7219793481587, 8231.42625961506, 823.142625961506, 821.8560936836235, 823.142625961506, 821.8560936836235, 110.5459599214397, 10.514083884078522, 8.944221663469648, 7.1371005126796945, 6, 6, 0.0, 110.5459599214397, 168.93220022743773, 24191.797919814, 13.59760263902831, 0.012773100982101502, 0.14534434050916928, -1.0203774345765206, 6776743.286347251, 58.30879404136442, 677674.3286347252, -0.26388408557763543, 14.856576661180434, 7.128866116404867, 8.944221663469648, 5.526921272838972, 5,  # altitude

            292.2058167861145, 6.273659084230401, 1899.1401259011516, 189.91401259011516, 237.3093354058276, 189.91401259011516, 237.3093354058276, 9185.583949441221, 95.84145214593329, 87.00469121848081, 46.16889140479205, 8, 7, 0.0, 9185.583949441221, -590681.5171580699, 167701545.96934026, 144.35396838001267, 0.5046570857980057, -0.6709547405719045, -1.0124250027307642, 452529.16127519653, 42.623977589485186, 45252.91612751965, -0.7359944832965826, 161.5861610371254, 84.18227197443197, 87.00469121848081, 40.19661248685052, 2,  # course
            14.280815369399086, 1.5223602512857481, 80.6168005299154, 8.06168005299154, 8.140671562447913, 8.06168005299154, 8.140671562447913, 17.25308447558101, 4.153683242085392, 3.6705735753874267, 3.175446598587203, 5, 5, 0.0, 17.25308447558101, -11.703736487886554, 537.5921503054634, 6.350893197174406, 0.5152379175038133, -0.1633143745039783, -1.1939930335248137, 822.4376975238267, 16.354100713026167, 82.24376975238268, 0.17343620112163832, 4.605977780767624, 2.4935447931400434, 3.6705735753874267, 1.9442155495851217, 1,  # speed

            25.332248443829698, 10.927633865659319, 178.6932721405494, 17.86932721405494, 16.813619393701735, 17.86932721405494, 16.813619393701735, 25.94684885631627, 5.09380494878988, 4.6304341966581575, 5.516079350886748, 4, 4, 0.0, 25.94684885631627, 14.536408525652845, 987.0793789054711, 9.40668135115098, 0.2850585748289056, 0.10998417125579417, -1.533835043154312, 3452.597039392807, 24.67435155531082, 345.2597039392807, 0.25153956646377806, 5.511423158443187, 3.164058334019472, 4.6304341966581575, 2.122717128289824, 5,  # hacc
            28.46768752255568, 11.348279679080182, 178.94939950216585, 17.894939950216585, 15.790523147120274, 17.894939950216585, 15.790523147120274, 32.598923648062375, 5.709546711260218, 5.009119172109466, 3.8255778695417275, 6, 6, 0.0, 32.598923648062375, 115.45118355182176, 2110.450750552408, 8.84916321773341, 0.3190592830791318, 0.6202881697793732, -1.014048215338768, 3528.2779946991996, 24.63449532424495, 352.82779946991997, -0.4737614184542243, 8.705562407934565, 4.650571531895659, 5.009119172109466, 2.740008899194992, 5,  # vacc

            # space features:
            1.2023130844274352,  # distance
            7.4855710351734,  # hcr
            0.8317301150192666,  # sr
            0.8729401569671781,  # trajectory

            # time features:
            1,  # month
            3,  # dayofweek
            12,  # hours
            720,  # minutes
            43200,  # seconds
        ]

        # act
        output = create_point(test_al, test_stamp, test_filename, test_person_stats, test_clusters)

        # assert
        self.assertEqual(output, expected_output)

    def test_create_point_withfilter_noperson_nogpsfeatures(self):
        """Test with lowpass filtering, no person features, and no gps features."""

        # arrange
        test_al = Mock(spec=AL)

        test_conf = Mock(spec=Config)
        test_conf.filter_data = True
        test_conf.fftfeatures = 0
        test_conf.local = 1
        test_conf.sfeatures = 0
        test_conf.gpsfeatures = 0
        test_conf.samplesize = 10

        test_al.conf = test_conf

        # Set data values:
        test_al.yaw = [0.3640372818564943, -2.322949529941176, 2.008221392962083, 4.907929027158476, 0.9122510136974293, -2.418333798810224, -3.1923166549519175, -4.175033262925778, -4.965486689737542, -1.3906230749752946]
        test_al.pitch = [-3.162968295986767, -4.666647948132239, 4.520716399063069, -3.154039655551234, 0.4182125473423346, -4.168318961881615, -2.5980476073613357, -1.6685651057525641, 3.2371457399291863, 4.194327681646078]
        test_al.roll = [3.9380543732275832, 4.9065900139325755, 1.559575817947608, 3.748158037105849, -1.7157110188361768, 4.875466947054095, -4.4476576053834185, -0.9891119438730014, 1.2696021092024594, 3.5692231813588506]

        test_al.rotx = [9.660077729122644, 1.6355327763219947, -8.94717268162079, -9.964566797618609, 7.897804661527129, -5.590226730125809, 0.3755598247586498, 3.3920453424214223, 3.891498031535436, -8.623797657305122]
        test_al.roty = [-2.0583837392105897, -3.7471621395043764, -9.6355809937429, 0.34834154289583097, 8.825831354570276, -9.995443704445035, 9.281055093155675, 8.263780472795656, 2.7874354315583183, 9.768673315076736]
        test_al.rotz = [2.447353180234547, -6.381022845232042, 8.110399171396288, -8.73097052902153, 0.13941149979300604, 9.928350186964845, 5.5885763896398455, -8.166959369809899, -5.800236031015102, 0.9586844281065137]

        test_al.accx = [-0.5344997627576218, -0.05963941376376036, -0.6846615580872408, 0.3129749488965732, -0.7512729528605864, -0.2289609070868066, 0.08833049044632557, 0.5395261728868226, 0.21064853543393292, 0.13327859486500748]
        test_al.accy = [-0.7787597015645251, 0.7368678712027816, -0.49432635588936247, 0.9604264402547806, -0.6839960928511306, 0.7196290960057685, 0.7853032115413037, 0.6614017016923766, 0.979414274628825, -0.6869694865247487]
        test_al.accz = [-0.24980654859071416, -0.5804666701154173, -0.5046045365473024, -0.279279706155791, 0.5399438139166117, 0.6843085375964921, 0.2215236755012131, -0.29441180878723716, -0.7834837503762135, 0.45188280622940913]
        test_al.acctotal = [0.9770158549827259, 0.9399322711769635, 0.9837406840999424, 1.0480312113900745, 1.15056552482781, 1.0191012253056821, 0.820716850324849, 0.9028947975178757, 1.2717986923765667, 0.832999597737618]

        test_al.latitude = [45.720049024287654, 45.16015407024762, 45.8461484021051, 45.36231411218953, 45.55199912734816, 45.9327322235666, 45.715660777520455, 45.313824413786, 45.67700785441802, 45.287249867092605]
        test_al.longitude = [-115.33785147412178, -115.8627325962175, -115.7079428458205, -115.74277058776053, -115.95217729730174, -115.99332137643903, -115.28416252514502, -115.14186050672829, -115.67576853665538, -115.07208321504989]
        test_al.altitude = [810.2213245992733, 828.020444363237, 840.4794306806942, 824.2303516812959, 819.4818356859511, 829.9659440293694, 818.1985404664574, 837.7380673702816, 806.7219793481587, 816.3683413903411]

        test_al.course = [225.6813948563143, 292.2058167861145, 6.273659084230401, 274.7506368351248, 126.60364340486748, 248.9372759553409, 261.9747954153205, 62.310980695282076, 270.95761178488016, 129.44431108367675]
        test_al.speed = [11.571468342724664, 14.280815369399086, 12.796382917195837, 6.714116530364987, 1.5223602512857481, 9.56722659453084, 6.422785860176224, 5.220575145550258, 10.445374918044397, 2.0756946006433474]

        test_al.hacc = [22.592997936520113, 25.332248443829698, 16.629675669378507, 10.927633865659319, 11.560839234746618, 14.409677626228634, 23.759149321008934, 13.538402787000809, 16.997563118024964, 22.94508413815179]
        test_al.vacc = [14.33151360551745, 13.064804053288855, 20.095012424826972, 26.14868844300875, 11.348279679080182, 21.913967271022265, 13.886057128399077, 17.249532688723097, 28.46768752255568, 12.443856685743517]

        test_al.minlat = 45.16015407024762
        test_al.maxlat = 45.9327322235666
        test_al.minlong = -115.99332137643903
        test_al.maxlong = -115.07208321504989

        test_stamp = datetime(2021, 1, 7, 12, 0, 0)

        test_filename = Mock(spec=str)
        test_person_stats = Mock(spec=np.ndarray)
        test_clusters = [Mock(spec=KMeans) for _ in range(5)]

        expected_output = [
            # statistical features for motion:
            4.138295362622016, -3.9665798696965453, -12.350513329042435, -1.2350513329042436, -1.7658358157212777, 2.7588708494333987, 3.081353787189892, 7.681971414399871, 2.771636955735702, 2.361046707327869, 2.15976556282084, 1, 1, 0.0, 7.681971414399871, 12.951124304767404, 120.15766163996473, 5.539475305762364, -2.244147171776376, 0.6082728629613154, -0.9638672257760073, 92.07323209308419, 7.6007334618321085, 9.207323209308418, 0.7748026537991487, 1.5501796354038992, 1.3946244377749748, 2.361046707327869, 1.4516989564700031, 1,  # yaw
            2.1587066952620306, -4.0115942987243205, -9.748778611699041, -0.9748778611699042, -1.085876095704645, 1.9540376444136762, 1.8875813625694078, 4.271509028902788, 2.066762934857984, 1.8702415065325724, 1.7763914126270872, 3, 3, 0.0, 4.271509028902788, 0.21613276245997462, 29.483292909251418, 3.5527828252541744, -2.1200224327360977, 0.024482096831201196, -1.3841042835280715, 52.218958731019946, 3.6514491575387975, 5.221895873101994, 0.600232092948012, 1.6031565794180078, 0.8863953847204233, 1.8702415065325724, 0.879605443449256, 1,  # pitch
            4.3404328173239985, -1.3203258880350206, 12.907383991281906, 1.2907383991281907, 1.130813181972056, 1.8308148519751732, 1.4954131597254463, 3.688924214945594, 1.9206572351530071, 1.6461759685011899, 1.3964626399572468, 1, 1, 0.0, 3.688924214945594, 1.7711283311373394, 25.7445942245087, 2.7929252799144937, 1.4880298257573226, 0.24997721791358027, -1.1081504994918245, 53.549298299296, 1.4860740321148243, 5.3549298299296, 1.0785275566415122, 0.8373519247596521, 0.64024704921926, 1.6461759685011899, 0.9894588903409598, 1,  # roll

            9.868302070372895, -8.94411225993247, 4.571138277378411, 0.45711382773784115, 0.634901215109975, 3.667797039190423, 2.421960173484078, 24.72543795501542, 4.972467994368131, 3.5177638049084097, 2.403212151962026, 4, 4, 0.0, 24.72543795501542, -17.657818692152272, 1773.3697067195262, 4.806424303924052, 10.87796450826223, -0.14362203100039989, -0.09924332905718991, 249.34391006524555, 6.192486782365534, 24.934391006524557, 0.5071401333429095, 3.981607751111151, 3.387133048415877, 3.5177638049084097, 3.51436705082035, 9,  # rotx
            9.484626374783243, -6.56010471962868, 16.883883380019704, 1.6883883380019704, 1.4448102129024203, 4.588777021118543, 4.283364164476298, 29.660202063743633, 5.4461180728794005, 4.475821283675172, 4.527776060756823, 3, 3, 0.0, 29.660202063743633, 0.2147528470684449, 1658.65206346262, 9.055552121513648, 3.2256311834777933, 0.0013294676438706265, -1.1145844588857023, 325.1085724364469, 7.098652536261894, 32.510857243644686, 0.8003633202760022, 2.8364827213335797, 2.0546861339694833, 4.475821283675172, 3.10277712708212, 1,  # roty
            7.463788223094775, -10.096098446165938, -13.30504349272785, -1.3305043492727848, -0.2804503729579084, 4.025149755890198, 3.543474280893153, 27.526583108770513, 5.246578228595331, 4.179070476561593, 3.2630239079352448, 3, 3, 0.0, 27.526583108770513, -16.801365045187215, 1741.4313374999551, 4.081788564550271, -3.943300321763667, -0.11633653101488511, -0.7017263151948256, 292.9682493220431, 5.1661053488719055, 29.29682493220431, 0.5683656563041232, 3.690052221049608, 2.7801065921397936, 4.179070476561593, 3.1720581742303815, 1,  # rotz

            0.4185283224026482, -0.44615349943326776, -0.6193070142899323, -0.06193070142899323, -0.20121661655614187, 0.3263690646117633, 0.3737971369202339, 0.11462430775586716, 0.3385621180165719, 0.3139829243259646, 0.2284183853900319, 3, 1, 0.0, 0.11462430775586716, 0.016072339197659286, 0.019277602706101956, 0.7475942738404678, -5.466789656899833, 0.41415604107364323, -1.5327653527758993, 1.184597195353543, -10.412616492816957, 0.11845971953535431, 0.8764540011881038, 0.1464230326027493, 0.13518707217862544, 0.3139829243259646, 0.12664529595521015, 1,  # accx
            0.8207233472629354, -0.4603187832722112, 3.202781035009409, 0.3202781035009409, 0.36323587108993527, 0.41571036876710893, 0.4242713044394436, 0.14110209990910033, 0.3756355945715213, 0.30795860643870776, 0.25399551701940304, 1, 3, 0.0, 0.14110209990910033, -0.02701234342339163, 0.050986227703318544, 0.4964268608283008, 1.172841947249815, -0.5096382865819544, -0.439139466581159, 2.436801634912598, -11.223420489279796, 0.24368016349125982, 0.9137160707243301, 0.25434061204298775, 0.17350326435909094, 0.30795860643870776, 0.21508974087442995, 1,  # accy
            0.6829413187227269, -0.6005901166011282, -1.5860433607859286, -0.15860433607859287, -0.30035152284324196, 0.4378795554316152, 0.5017167261380231, 0.19322791644191417, 0.4395769744219028, 0.37443782100017814, 0.24244911645674527, 3, 2, 0.0, 0.19322791644191417, 0.06777396236488284, 0.08009921332765113, 0.7467223542038489, -2.7715318842486134, 0.797917648033262, -0.854697647939004, 2.1838325186484537, -7.9030706789458876, 0.21838325186484536, 0.7026601375619358, 0.314696743093571, 0.2037995990157159, 0.37443782100017814, 0.23026991693782475, 1,  # accz
            1.1346296521132564, 0.8667295315050305, 10.074336485175523, 1.0074336485175523, 0.9806400695585245, 1.0074336485175523, 0.9806400695585245, 0.006621193192848626, 0.08137071458976274, 0.07255604399304878, 0.06213406490593271, 3, 3, 0.0, 0.006621193192848626, 2.811934520723419e-05, 8.262024523777219e-05, 0.13675011624065614, 0.08077029659422283, 0.05219162435386405, -1.1154226814069514, 10.21543749358236, 0.03589796890804258, 1.021543749358236, 0.37551722577181135, 0.08106593225093021, 0.045771836867694364, 0.07255604399304878, 0.03683495178397002, 5,  # acctotal

            # 2D features:
            -0.0014112361083133225, -0.32241190889823623, 0.2581391624287407, -2.698649420740387, -0.17007890569968714, -2.376237511842151, 0.6966505372389256, -0.31732668105234235, -0.2248167500814314, -0.017988166469840472, -0.29614427705120416, 0.2993385145825019, 0.3006830799334257, -2.011179213413264, -0.19086412063677605, -0.43668315820395287, -0.38290114982029205, 1.5744960552093117,

            # GPS statistical features:
            45.9327322235666, 45.16015407024762, 455.56713987256177, 45.556713987256174, 45.614503490883095, 45.556713987256174, 45.614503490883095, 0.061825459153484884, 0.24864725848777194, 0.22160566912339164, 0.24191714495778527, 7, 7, 0.0, 0.061825459153484884, -0.0016079471624167695, 0.006347986581188151, 0.4062246105016527, 0.005457971761469174, -0.10459736597507543, -1.3392613260619854, 20754.760147758163, 33.17091838978416, 2075.476014775816, -0.5205399522048085, 0.40799899165868836, 0.14656878549018248, 0.22160566912339164, 0.11276695688834894, 3,  # latitude
            -115.07208321504989, -115.99332137643903, -1155.7706709612396, -115.57706709612395, -115.69185569123795, 115.57706709612395, 115.69185569123795, 0.1037338847322395, 0.3220774514495535, 0.2944621326901796, 0.2808936456324389, 2, 4, 0.0, 0.1037338847322395, 0.009405457347791843, 0.016916053846571667, 0.5785700710724768, -0.002786689951058248, 0.28151332423737574, -1.4279810581881984, 133581.62172426673, 41.257399647154436, 13358.162172426673, 0.13806656369860437, 0.32823373603939704, 0.24703361433556775, 0.2944621326901796, 0.1304834746003897, 2,  # longitude
            840.4794306806942, 806.7219793481587, 8231.42625961506, 823.142625961506, 821.8560936836235, 823.142625961506, 821.8560936836235, 110.5459599214397, 10.514083884078522, 8.944221663469648, 7.1371005126796945, 6, 6, 0.0, 110.5459599214397, 168.93220022743773, 24191.797919814, 13.59760263902831, 0.012773100982101502, 0.14534434050916928, -1.0203774345765206, 6776743.286347251, 58.30879404136442, 677674.3286347252, -0.26388408557763543, 14.856576661180434, 7.128866116404867, 8.944221663469648, 5.526921272838972, 5,  # altitude

            292.2058167861145, 6.273659084230401, 1899.1401259011516, 189.91401259011516, 237.3093354058276, 189.91401259011516, 237.3093354058276, 9185.583949441221, 95.84145214593329, 87.00469121848081, 46.16889140479205, 8, 7, 0.0, 9185.583949441221, -590681.5171580699, 167701545.96934026, 144.35396838001267, 0.5046570857980057, -0.6709547405719045, -1.0124250027307642, 452529.16127519653, 42.623977589485186, 45252.91612751965, -0.7359944832965826, 161.5861610371254, 84.18227197443197, 87.00469121848081, 40.19661248685052, 2,  # course
            14.280815369399086, 1.5223602512857481, 80.6168005299154, 8.06168005299154, 8.140671562447913, 8.06168005299154, 8.140671562447913, 17.25308447558101, 4.153683242085392, 3.6705735753874267, 3.175446598587203, 5, 5, 0.0, 17.25308447558101, -11.703736487886554, 537.5921503054634, 6.350893197174406, 0.5152379175038133, -0.1633143745039783, -1.1939930335248137, 822.4376975238267, 16.354100713026167, 82.24376975238268, 0.17343620112163832, 4.605977780767624, 2.4935447931400434, 3.6705735753874267, 1.9442155495851217, 1,  # speed

            25.332248443829698, 10.927633865659319, 178.6932721405494, 17.86932721405494, 16.813619393701735, 17.86932721405494, 16.813619393701735, 25.94684885631627, 5.09380494878988, 4.6304341966581575, 5.516079350886748, 4, 4, 0.0, 25.94684885631627, 14.536408525652845, 987.0793789054711, 9.40668135115098, 0.2850585748289056, 0.10998417125579417, -1.533835043154312, 3452.597039392807, 24.67435155531082, 345.2597039392807, 0.25153956646377806, 5.511423158443187, 3.164058334019472, 4.6304341966581575, 2.122717128289824, 5,  # hacc
            28.46768752255568, 11.348279679080182, 178.94939950216585, 17.894939950216585, 15.790523147120274, 17.894939950216585, 15.790523147120274, 32.598923648062375, 5.709546711260218, 5.009119172109466, 3.8255778695417275, 6, 6, 0.0, 32.598923648062375, 115.45118355182176, 2110.450750552408, 8.84916321773341, 0.3190592830791318, 0.6202881697793732, -1.014048215338768, 3528.2779946991996, 24.63449532424495, 352.82779946991997, -0.4737614184542243, 8.705562407934565, 4.650571531895659, 5.009119172109466, 2.740008899194992, 5,  # vacc

            # space features:
            1.2023130844274352,  # distance
            7.4855710351734,  # hcr
            0.8317301150192666,  # sr
            0.8729401569671781,  # trajectory

            # time features:
            1,  # month
            3,  # dayofweek
            12,  # hours
            720,  # minutes
            43200,  # seconds
        ]

        # act
        output = create_point(test_al, test_stamp, test_filename, test_person_stats, test_clusters)

        # assert
        self.assertEqual(output, expected_output)
