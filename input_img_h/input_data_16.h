#include <tvmgen_default.h>
const size_t input_len = 784;
const static __attribute__((aligned(16))) float input_data[] = {0.9764705896377563, 0.9686274528503418, 0.9686274528503418, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.11372549086809158, -0.4901960790157318, 0.9921568632125854, 0.9764705896377563, 1.0, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.13725490868091583, 0.03529411926865578, 0.686274528503418, 0.364705890417099, 0.7176470756530762, 0.9607843160629272, 0.46666666865348816, 0.9058823585510254, 0.7098039388656616, 0.9450980424880981, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9843137264251709, 0.12156862765550613, 1.0, 0.9764705896377563, 0.9764705896377563, 0.9921568632125854, 0.9921568632125854, 0.9686274528503418, 0.9686274528503418, 0.11372549086809158, 0.003921568859368563, 0.6784313917160034, 0.239215686917305, -0.3960784375667572, -0.40392157435417175, 0.9686274528503418, 0.9764705896377563, 0.9607843160629272, 0.5529412031173706, 0.9764705896377563, 0.9764705896377563, 0.8666666746139526, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.2235294133424759, 0.06666667014360428, 0.027450980618596077, 0.03529411926865578, 0.05882352963089943, 0.15294118225574493, 0.239215686917305, 0.3176470696926117, -0.2862745225429535, -0.26274511218070984, -0.45098039507865906, 0.0117647061124444, -0.07450980693101883, 0.2705882489681244, -0.13725490868091583, -0.45098039507865906, 0.45098039507865906, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.8274509906768799, 0.9764705896377563, 0.35686275362968445, 0.5215686559677124, 0.9607843160629272, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.003921568859368563, -0.11372549086809158, -0.10588235408067703, -0.13725490868091583, -0.12941177189350128, -0.08235294371843338, -0.5843137502670288, -0.27843138575553894, 0.3490196168422699, 0.41960784792900085, -0.37254902720451355, -0.18431372940540314, -0.08235294371843338, -0.019607843831181526, 0.7647058963775635, 0.6078431606292725, 0.843137264251709, 0.9921568632125854, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9843137264251709, -0.5372549295425415, 0.16078431904315948, -0.30980393290519714, -0.239215686917305, -0.11372549086809158, -0.09803921729326248, -0.05098039284348488, -0.239215686917305, -0.26274511218070984, -0.27843138575553894, -0.27843138575553894, -0.24705882370471954, 0.4431372582912445, 0.14509804546833038, 0.14509804546833038, -0.08235294371843338, 0.05882352963089943, -0.7019608020782471, -0.6784313917160034, -0.8196078538894653, 0.843137264251709, 0.37254902720451355, 0.686274528503418, 0.9686274528503418, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.05882352963089943, -0.6784313917160034, -0.45098039507865906, -0.43529412150382996, -0.37254902720451355, -0.6705882549285889, -0.08235294371843338, -0.27843138575553894, -0.3490196168422699, -0.3803921639919281, -0.35686275362968445, -0.7019608020782471, 0.5372549295425415, 0.5607843399047852, 0.6470588445663452, -0.2549019753932953, -0.05098039284348488, 0.11372549086809158, -0.0117647061124444, -0.6784313917160034, 0.2705882489681244, 0.2549019753932953, 0.32549020648002625, 0.12941177189350128, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.4588235318660736, -0.4274509847164154, -0.5215686559677124, -0.5372549295425415, -0.5607843399047852, 0.26274511218070984, -0.10588235408067703, -0.32549020648002625, -0.40392157435417175, -0.4117647111415863, -0.4274509847164154, -0.6941176652908325, 0.5921568870544434, 0.5843137502670288, -0.5215686559677124, -0.027450980618596077, 0.1921568661928177, 0.364705890417099, 0.4588235318660736, -0.0117647061124444, -0.4117647111415863, 0.5843137502670288, -0.26274511218070984, 0.6000000238418579, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.545098066329956, -0.48235294222831726, -0.6078431606292725, -0.615686297416687, -0.615686297416687, 0.18431372940540314, -0.1764705926179886, -0.364705890417099, -0.4274509847164154, -0.4431372582912445, -0.46666666865348816, -0.5686274766921997, 0.6470588445663452, 0.43529412150382996, -0.1921568661928177, 0.14509804546833038, 0.37254902720451355, 0.5215686559677124, 0.6705882549285889, 0.6470588445663452, -0.6549019813537598, 0.8274509906768799, 0.5843137502670288, 0.7411764860153198, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.6313725709915161, -0.5372549295425415, -0.6705882549285889, -0.615686297416687, -0.5921568870544434, 0.10588235408067703, -0.1921568661928177, -0.4274509847164154, -0.4588235318660736, -0.48235294222831726, -0.5137255191802979, -0.5137255191802979, 0.686274528503418, 0.12941177189350128, 0.03529411926865578, 0.29411765933036804, 0.364705890417099, 0.545098066329956, 0.6627451181411743, 0.7882353067398071, -0.6078431606292725, 0.8274509906768799, 0.3960784375667572, 0.7490196228027344, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.5607843399047852, -0.6392157077789307, -0.8745098114013672, -0.6549019813537598, -0.6549019813537598, 0.07450980693101883, -0.26274511218070984, -0.41960784792900085, -0.48235294222831726, -0.49803921580314636, -0.1764705926179886, -0.30980393290519714, -0.46666666865348816, -0.3333333432674408, 0.10588235408067703, 0.2705882489681244, 0.4117647111415863, 0.6000000238418579, 0.7176470756530762, 0.7960784435272217, -0.5607843399047852, 0.8274509906768799, 0.2235294133424759, -0.18431372940540314, 0.9607843160629272, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.5843137502670288, -0.6392157077789307, -0.8666666746139526, -0.7019608020782471, -0.6470588445663452, 0.05098039284348488, -0.23137255012989044, -0.4274509847164154, -0.4745098054409027, -0.5058823823928833, -0.7411764860153198, -0.2235294133424759, -0.48235294222831726, -0.08235294371843338, -0.06666667014360428, 0.41960784792900085, 0.3960784375667572, -0.07450980693101883, 0.7647058963775635, 0.3333333432674408, -0.5137255191802979, 0.4901960790157318, -0.14509804546833038, -0.05882352963089943, 0.003921568859368563, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.5137255191802979, -0.545098066329956, -0.8039215803146362, -0.7019608020782471, -0.686274528503418, 0.019607843831181526, -0.2549019753932953, -0.40392157435417175, -0.45098039507865906, -0.48235294222831726, -0.7019608020782471, -0.35686275362968445, -0.7882353067398071, -0.45098039507865906, 0.23137255012989044, -0.003921568859368563, 0.37254902720451355, 0.0117647061124444, 0.6470588445663452, -0.29411765933036804, -0.43529412150382996, 0.05882352963089943, 0.7960784435272217, -0.35686275362968445, 0.06666667014360428, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.5058823823928833, -0.6000000238418579, -0.7254902124404907, -0.7882353067398071, -0.7098039388656616, -0.615686297416687, -0.20000000298023224, -0.32549020648002625, -0.38823530077934265, -0.43529412150382996, -0.5215686559677124, -0.0117647061124444, -0.7019608020782471, -0.40392157435417175, 0.14509804546833038, 0.11372549086809158, 0.10588235408067703, 0.12156862765550613, 0.3019607961177826, 0.772549033164978, -0.3176470696926117, -0.1764705926179886, 0.7647058963775635, 0.6078431606292725, -0.09803921729326248, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.615686297416687, -0.6313725709915161, -0.7882353067398071, 0.545098066329956, 0.14509804546833038, -0.08235294371843338, -0.45098039507865906, -0.5137255191802979, -0.529411792755127, -0.6313725709915161, -0.686274528503418, 0.7254902124404907, -0.545098066329956, -0.24705882370471954, 0.21568627655506134, 0.3803921639919281, 0.3490196168422699, 0.2705882489681244, 0.1764705926179886, 0.8509804010391235, -0.5058823823928833, -0.3019607961177826, 0.7333333492279053, 0.38823530077934265, 0.7568627595901489, 0.9764705896377563, 0.9764705896377563, 0.9764705896377563, -0.6941176652908325, -0.6941176652908325, 0.09019608050584793, -0.04313725605607033, -0.15294118225574493, -0.16078431904315948, -0.27843138575553894, -0.364705890417099, -0.49803921580314636, -0.5921568870544434, -0.6784313917160034, 0.364705890417099, 0.7019608020782471, 0.21568627655506134, 0.24705882370471954, 0.37254902720451355, 0.46666666865348816, 0.03529411926865578, 0.30980393290519714, 0.8745098114013672, -0.5529412031173706, -0.37254902720451355, 0.5607843399047852, 0.1921568661928177, 0.2862745225429535, 0.9450980424880981, 0.9764705896377563, 0.9764705896377563, -0.6000000238418579, 0.08235294371843338, -0.12941177189350128, -0.21568627655506134, -0.23137255012989044, -0.2705882489681244, -0.364705890417099, -0.37254902720451355, -0.46666666865348816, -0.6078431606292725, -0.7176470756530762, -0.3019607961177826, 0.5843137502670288, 0.5764706134796143, 0.1764705926179886, 0.30980393290519714, 0.40392157435417175, 0.3333333432674408, 0.4117647111415863, 0.239215686917305, -0.6000000238418579, -0.5137255191802979, 0.30980393290519714, 0.11372549086809158, -0.04313725605607033, 0.5686274766921997, 0.08235294371843338, 0.9764705896377563, -0.6392157077789307, -0.10588235408067703, -0.2549019753932953, -0.34117648005485535, -0.2549019753932953, -0.27843138575553894, -0.3803921639919281, -0.40392157435417175, -0.529411792755127, -0.6470588445663452, -0.8196078538894653, -0.24705882370471954, -0.09803921729326248, -0.27843138575553894, 0.18431372940540314, 0.29411765933036804, 0.18431372940540314, 0.3490196168422699, 0.2549019753932953, -0.5372549295425415, -0.5529412031173706, -0.615686297416687, 0.08235294371843338, -0.23137255012989044, -0.3019607961177826, 0.04313725605607033, 0.5215686559677124, -0.05882352963089943, -0.4274509847164154, -0.20000000298023224, -0.32549020648002625, -0.35686275362968445, -0.3490196168422699, -0.34117648005485535, -0.3803921639919281, -0.46666666865348816, -0.5764706134796143, -0.7490196228027344, -0.8274509906768799, -0.2078431397676468, 0.0117647061124444, -0.3803921639919281, 0.13725490868091583, 0.30980393290519714, 0.21568627655506134, -0.34117648005485535, 0.0117647061124444, -0.37254902720451355, -0.5372549295425415, -0.5058823823928833, -0.04313725605607033, -0.4588235318660736, -0.41960784792900085, -0.2549019753932953, 0.019607843831181526, 0.4117647111415863, -0.019607843831181526, -0.239215686917305, -0.35686275362968445, -0.364705890417099, -0.364705890417099, -0.38823530077934265, -0.48235294222831726, -0.5607843399047852, -0.6627451181411743, -0.7882353067398071, -0.8117647171020508, -0.8117647171020508, -0.45098039507865906, -0.7803921699523926, -0.4901960790157318, 0.16078431904315948, 0.26274511218070984, 0.08235294371843338, 0.6784313917160034, 0.8509804010391235, 0.8823529481887817, 0.6941176652908325, -0.364705890417099, -0.615686297416687, -0.49803921580314636, -0.3803921639919281, -0.2078431397676468, -0.05882352963089943, 0.003921568859368563, -0.30980393290519714, -0.364705890417099, -0.37254902720451355, -0.40392157435417175, -0.40392157435417175, -0.545098066329956, -0.6235294342041016, -0.7490196228027344, -0.8196078538894653, -0.9058823585510254, -0.9215686321258545, -0.929411768913269, -0.7490196228027344, -0.772549033164978, 0.05882352963089943, 0.35686275362968445, 0.4901960790157318, 0.6313725709915161, 0.8823529481887817, 0.8823529481887817, 0.5607843399047852, 0.7568627595901489, -0.46666666865348816, -0.12156862765550613, -0.05098039284348488, -0.364705890417099, -0.27843138575553894, 0.03529411926865578, -0.29411765933036804, -0.3803921639919281, -0.38823530077934265, -0.4431372582912445, -0.4901960790157318, -0.5921568870544434, -0.7019608020782471, -0.7411764860153198, -0.7647058963775635, -0.7882353067398071, -0.7803921699523926, -0.7803921699523926, -0.7803921699523926, -0.7411764860153198, -0.09803921729326248, -0.8039215803146362, 0.7960784435272217, 0.929411768913269, 0.7176470756530762, 0.37254902720451355, 0.8196078538894653, 0.8196078538894653, 0.929411768913269, -0.3490196168422699, -0.07450980693101883, -0.03529411926865578, -0.364705890417099, 0.08235294371843338, -0.3176470696926117, -0.38823530077934265, -0.4117647111415863, -0.4431372582912445, -0.529411792755127, -0.6235294342041016, 0.29411765933036804, 0.2549019753932953, 0.38823530077934265, 0.35686275362968445, -0.4431372582912445, -0.8039215803146362, -0.3490196168422699, -0.41960784792900085, -0.7411764860153198, -0.8823529481887817, -0.8117647171020508, 0.239215686917305, 0.35686275362968445, 0.35686275362968445, 0.4901960790157318, 0.7803921699523926, 0.686274528503418, 0.9372549057006836, -0.12941177189350128, -0.1764705926179886, -0.03529411926865578, 0.10588235408067703, -0.3490196168422699, -0.4274509847164154, -0.4274509847164154, -0.48235294222831726, -0.5529412031173706, 0.4431372582912445, 0.3333333432674408, 0.3803921639919281, 0.3803921639919281, 0.48235294222831726, 0.4588235318660736, 0.1764705926179886, 0.15294118225574493, 0.7176470756530762, -0.772549033164978, -0.8274509906768799, -0.6313725709915161, -0.027450980618596077, 0.5607843399047852, 0.16862745583057404, 0.5764706134796143, 0.686274528503418, 0.5764706134796143, 0.7333333492279053, 0.7647058963775635, -0.32549020648002625, -0.27843138575553894, 0.12156862765550613, -0.364705890417099, -0.4274509847164154, -0.4431372582912445, -0.4901960790157318, -0.5137255191802979, 0.41960784792900085, 0.40392157435417175, 0.3803921639919281, 0.4588235318660736, 0.4588235318660736, 0.35686275362968445, 0.08235294371843338, 0.13725490868091583, -0.7490196228027344, -0.8588235378265381, -0.843137264251709, -0.8274509906768799, 0.6078431606292725, 0.6313725709915161, 0.5607843399047852, 0.10588235408067703, 0.45098039507865906, 0.3333333432674408, 0.5686274766921997, 0.8039215803146362, -0.43529412150382996, -0.45098039507865906, 0.15294118225574493, -0.3490196168422699, -0.4431372582912445, -0.43529412150382996, -0.48235294222831726, -0.5921568870544434, 0.45098039507865906, 0.4274509847164154, 0.4431372582912445, 0.5607843399047852, 0.5607843399047852, -0.06666667014360428, 0.23137255012989044, 0.3176470696926117, -0.7098039388656616, -0.9372549057006836, -0.8823529481887817, -0.7333333492279053, 0.3803921639919281, 0.6000000238418579, 0.38823530077934265, 0.2078431397676468, 0.06666667014360428, 0.26274511218070984, 0.364705890417099, 0.6627451181411743, 0.9450980424880981, -0.48235294222831726, 0.09019608050584793, -0.35686275362968445, -0.4745098054409027, -0.4431372582912445, -0.4901960790157318, -0.5215686559677124, 0.45098039507865906, 0.5686274766921997, 0.6235294342041016, 0.5058823823928833, 0.6078431606292725, 0.5843137502670288, 0.03529411926865578, -0.8117647171020508, -0.615686297416687, -0.6705882549285889, -0.8823529481887817, 0.09803921729326248, 0.8588235378265381, 0.6235294342041016, 0.239215686917305, 0.2235294133424759, 0.03529411926865578, 0.2078431397676468, 0.23137255012989044, 0.4745098054409027, 0.7019608020782471, 0.9529411792755127, 0.08235294371843338, -0.3960784375667572, -0.46666666865348816, -0.45098039507865906, -0.49803921580314636, -0.529411792755127, 0.48235294222831726, 0.5215686559677124, 0.5529412031173706, 0.6235294342041016, 0.6705882549285889, 0.545098066329956, -0.15294118225574493, -0.7568627595901489, -0.9215686321258545, -0.6470588445663452, -0.6627451181411743, 0.29411765933036804, 0.3333333432674408, 0.4588235318660736, 0.38823530077934265, 0.10588235408067703, -0.09019608050584793, 0.1764705926179886, 0.23137255012989044, 0.34117648005485535, 0.48235294222831726, 0.7176470756530762, 0.08235294371843338, -0.38823530077934265, -0.46666666865348816, -0.4274509847164154, -0.4745098054409027, -0.5529412031173706, 0.4745098054409027, 0.5372549295425415, 0.6549019813537598, 0.6627451181411743, 0.3803921639919281, 0.48235294222831726, 0.4117647111415863, -0.772549033164978, -0.8666666746139526, -0.6000000238418579, -0.7098039388656616, -0.7176470756530762, -0.6078431606292725, 0.3960784375667572, -0.11372549086809158, -0.027450980618596077, -0.6470588445663452, 0.11372549086809158, 0.21568627655506134, 0.239215686917305, 0.3333333432674408, 0.5058823823928833};
