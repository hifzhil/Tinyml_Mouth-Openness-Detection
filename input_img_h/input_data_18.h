#include <tvmgen_default.h>
const size_t input_len = 784;
const static __attribute__((aligned(16))) float input_data[] = {0.364705890417099, 0.27843138575553894, 0.4588235318660736, 0.5921568870544434, 0.5843137502670288, 0.4901960790157318, 0.40392157435417175, 0.12156862765550613, 0.14509804546833038, 0.07450980693101883, -0.27843138575553894, -0.37254902720451355, -0.43529412150382996, -0.3176470696926117, -0.3019607961177826, -0.16862745583057404, -0.16078431904315948, -0.15294118225574493, -0.12941177189350128, 0.06666667014360428, 0.1921568661928177, 0.2862745225429535, 0.29411765933036804, 0.18431372940540314, 0.03529411926865578, 0.10588235408067703, 0.18431372940540314, 0.29411765933036804, 0.38823530077934265, 0.3960784375667572, 0.5843137502670288, 0.615686297416687, 0.545098066329956, 0.4745098054409027, 0.27843138575553894, 0.20000000298023224, 0.07450980693101883, 0.16078431904315948, -0.09803921729326248, -0.38823530077934265, -0.48235294222831726, -0.4588235318660736, -0.41960784792900085, -0.2862745225429535, -0.21568627655506134, -0.18431372940540314, -0.09019608050584793, -0.027450980618596077, 0.1764705926179886, 0.24705882370471954, 0.26274511218070984, 0.239215686917305, 0.18431372940540314, 0.13725490868091583, 0.16862745583057404, 0.26274511218070984, 0.41960784792900085, 0.5058823823928833, 0.5764706134796143, 0.5607843399047852, 0.46666666865348816, 0.4117647111415863, 0.27843138575553894, 0.2078431397676468, 0.019607843831181526, 0.10588235408067703, -0.13725490868091583, -0.40392157435417175, -0.5058823823928833, -0.40392157435417175, -0.27843138575553894, -0.13725490868091583, -0.09019608050584793, -0.18431372940540314, -0.12941177189350128, -0.05882352963089943, 0.09803921729326248, 0.1921568661928177, 0.1921568661928177, 0.20000000298023224, 0.1764705926179886, 0.15294118225574493, 0.16862745583057404, 0.2549019753932953, 0.4901960790157318, 0.4745098054409027, 0.5137255191802979, 0.4901960790157318, 0.38823530077934265, 0.3490196168422699, 0.30980393290519714, 0.16078431904315948, 0.019607843831181526, 0.11372549086809158, -0.239215686917305, -0.43529412150382996, -0.3960784375667572, -0.27843138575553894, -0.15294118225574493, -0.09803921729326248, 0.003921568859368563, -0.09019608050584793, -0.06666667014360428, 0.12941177189350128, 0.11372549086809158, 0.18431372940540314, 0.16862745583057404, 0.1764705926179886, 0.10588235408067703, 0.16078431904315948, 0.14509804546833038, 0.239215686917305, 0.48235294222831726, 0.4117647111415863, 0.4588235318660736, 0.37254902720451355, 0.2235294133424759, 0.15294118225574493, 0.29411765933036804, 0.1921568661928177, 0.13725490868091583, 0.1921568661928177, -0.019607843831181526, -0.38823530077934265, -0.15294118225574493, -0.14509804546833038, 0.0117647061124444, 0.027450980618596077, -0.03529411926865578, -0.019607843831181526, 0.027450980618596077, 0.05098039284348488, 0.07450980693101883, 0.12156862765550613, 0.09803921729326248, 0.15294118225574493, 0.12941177189350128, 0.12156862765550613, 0.07450980693101883, 0.12156862765550613, 0.41960784792900085, 0.3803921639919281, 0.3803921639919281, 0.3333333432674408, 0.13725490868091583, 0.07450980693101883, 0.30980393290519714, 0.3333333432674408, 0.3019607961177826, 0.2705882489681244, 0.13725490868091583, -0.14509804546833038, -0.07450980693101883, -0.003921568859368563, 0.16862745583057404, 0.16078431904315948, 0.09803921729326248, 0.07450980693101883, 0.13725490868091583, 0.10588235408067703, 0.16078431904315948, 0.16078431904315948, 0.003921568859368563, 0.05882352963089943, 0.05882352963089943, 0.09803921729326248, 0.05882352963089943, 0.07450980693101883, 0.32549020648002625, 0.3019607961177826, 0.15294118225574493, 0.23137255012989044, 0.21568627655506134, 0.4274509847164154, 0.545098066329956, 0.48235294222831726, 0.46666666865348816, 0.43529412150382996, 0.26274511218070984, 0.12156862765550613, 0.09019608050584793, 0.13725490868091583, 0.3019607961177826, 0.3333333432674408, 0.23137255012989044, 0.30980393290519714, 0.2862745225429535, 0.24705882370471954, 0.24705882370471954, 0.239215686917305, 0.09019608050584793, 0.027450980618596077, -0.09019608050584793, 0.0117647061124444, -0.10588235408067703, 0.003921568859368563, 0.10588235408067703, 0.07450980693101883, 0.03529411926865578, 0.27843138575553894, 0.4274509847164154, 0.4901960790157318, 0.45098039507865906, 0.41960784792900085, 0.3019607961177826, 0.20000000298023224, 0.12156862765550613, 0.2705882489681244, 0.29411765933036804, 0.24705882370471954, 0.2862745225429535, 0.16078431904315948, 0.07450980693101883, 0.09803921729326248, 0.14509804546833038, 0.21568627655506134, 0.29411765933036804, 0.3176470696926117, 0.18431372940540314, -0.003921568859368563, -0.13725490868091583, -0.03529411926865578, -0.16078431904315948, -0.09803921729326248, 0.06666667014360428, 0.11372549086809158, 0.1764705926179886, 0.3176470696926117, 0.4117647111415863, 0.2549019753932953, 0.05882352963089943, 0.05882352963089943, 0.0117647061124444, -0.05882352963089943, -0.08235294371843338, -0.11372549086809158, 0.003921568859368563, -0.0117647061124444, -0.09019608050584793, -0.12941177189350128, -0.16078431904315948, -0.12941177189350128, -0.1921568661928177, -0.2235294133424759, -0.1764705926179886, -0.11372549086809158, -0.05098039284348488, -0.18431372940540314, -0.3490196168422699, -0.239215686917305, -0.11372549086809158, -0.12156862765550613, 0.09803921729326248, 0.09803921729326248, -0.019607843831181526, 0.11372549086809158, 0.019607843831181526, -0.16862745583057404, -0.23137255012989044, -0.38823530077934265, -0.37254902720451355, -0.27843138575553894, -0.08235294371843338, -0.11372549086809158, -0.2862745225429535, -0.35686275362968445, -0.21568627655506134, -0.21568627655506134, -0.34117648005485535, -0.4588235318660736, -0.41960784792900085, -0.4117647111415863, -0.4588235318660736, -0.49803921580314636, -0.6784313917160034, -0.6784313917160034, -0.6235294342041016, -0.3803921639919281, -0.2235294133424759, -0.18431372940540314, 0.019607843831181526, -0.20000000298023224, -0.364705890417099, -0.43529412150382996, -0.5686274766921997, -0.5686274766921997, -0.3333333432674408, -0.18431372940540314, -0.21568627655506134, -0.21568627655506134, -0.2549019753932953, -0.3490196168422699, -0.45098039507865906, -0.4431372582912445, -0.3960784375667572, -0.2862745225429535, -0.14509804546833038, -0.09803921729326248, -0.10588235408067703, -0.08235294371843338, -0.07450980693101883, -0.2235294133424759, -0.1921568661928177, -0.07450980693101883, -0.3176470696926117, -0.35686275362968445, -0.12156862765550613, -0.003921568859368563, -0.0117647061124444, -0.3176470696926117, -0.49803921580314636, -0.2235294133424759, -0.08235294371843338, 0.09803921729326248, 0.29411765933036804, 0.29411765933036804, 0.24705882370471954, 0.14509804546833038, 0.11372549086809158, 0.11372549086809158, 0.12941177189350128, 0.04313725605607033, 0.10588235408067703, 0.16862745583057404, 0.16862745583057404, 0.10588235408067703, 0.15294118225574493, 0.06666667014360428, -0.019607843831181526, 0.04313725605607033, 0.14509804546833038, 0.12156862765550613, -0.07450980693101883, -0.3333333432674408, -0.1921568661928177, 0.019607843831181526, 0.0117647061124444, -0.20000000298023224, -0.027450980618596077, 0.4117647111415863, 0.37254902720451355, 0.364705890417099, 0.3019607961177826, 0.364705890417099, 0.4274509847164154, 0.4117647111415863, 0.37254902720451355, 0.30980393290519714, 0.26274511218070984, 0.13725490868091583, 0.12941177189350128, 0.239215686917305, 0.21568627655506134, 0.1764705926179886, 0.18431372940540314, 0.2235294133424759, 0.29411765933036804, 0.32549020648002625, 0.23137255012989044, 0.20000000298023224, 0.06666667014360428, -0.13725490868091583, -0.05882352963089943, 0.09803921729326248, -0.05882352963089943, 0.027450980618596077, 0.2862745225429535, 0.40392157435417175, 0.545098066329956, 0.5764706134796143, 0.5921568870544434, 0.6313725709915161, 0.6000000238418579, 0.529411792755127, 0.5058823823928833, 0.4274509847164154, 0.37254902720451355, 0.24705882370471954, 0.3019607961177826, 0.27843138575553894, 0.30980393290519714, 0.30980393290519714, 0.4117647111415863, 0.43529412150382996, 0.40392157435417175, 0.2862745225429535, 0.26274511218070984, 0.239215686917305, 0.2549019753932953, 0.09803921729326248, -0.09019608050584793, -0.019607843831181526, 0.09019608050584793, 0.11372549086809158, 0.4117647111415863, 0.5215686559677124, 0.529411792755127, 0.48235294222831726, 0.49803921580314636, 0.5372549295425415, 0.5215686559677124, 0.48235294222831726, 0.34117648005485535, 0.20000000298023224, 0.13725490868091583, 0.05882352963089943, 0.027450980618596077, 0.05882352963089943, 0.14509804546833038, 0.23137255012989044, 0.364705890417099, 0.35686275362968445, 0.34117648005485535, 0.2862745225429535, 0.2862745225429535, 0.3019607961177826, 0.30980393290519714, 0.18431372940540314, -0.05098039284348488, -0.019607843831181526, 0.08235294371843338, 0.35686275362968445, 0.5372549295425415, 0.5137255191802979, 0.48235294222831726, 0.41960784792900085, 0.4274509847164154, 0.4274509847164154, 0.37254902720451355, 0.24705882370471954, -0.003921568859368563, -0.12941177189350128, -0.23137255012989044, -0.2549019753932953, -0.2078431397676468, -0.08235294371843338, 0.027450980618596077, 0.04313725605607033, 0.11372549086809158, 0.16862745583057404, 0.21568627655506134, 0.26274511218070984, 0.3019607961177826, 0.32549020648002625, 0.364705890417099, 0.2549019753932953, 0.05098039284348488, -0.04313725605607033, 0.15294118225574493, 0.38823530077934265, 0.5058823823928833, 0.529411792755127, 0.529411792755127, 0.4274509847164154, 0.3333333432674408, 0.29411765933036804, 0.2078431397676468, 0.09019608050584793, -0.05098039284348488, -0.12156862765550613, -0.11372549086809158, -0.04313725605607033, -0.10588235408067703, -0.027450980618596077, -0.0117647061124444, -0.027450980618596077, 0.04313725605607033, 0.16862745583057404, 0.26274511218070984, 0.3333333432674408, 0.34117648005485535, 0.34117648005485535, 0.27843138575553894, 0.2235294133424759, 0.09019608050584793, -0.003921568859368563, 0.18431372940540314, 0.364705890417099, 0.5215686559677124, 0.5372549295425415, 0.4745098054409027, 0.46666666865348816, 0.3960784375667572, 0.30980393290519714, 0.20000000298023224, 0.09803921729326248, 0.003921568859368563, -0.05098039284348488, -0.03529411926865578, -0.09803921729326248, -0.027450980618596077, 0.05098039284348488, 0.05882352963089943, 0.05882352963089943, 0.11372549086809158, 0.239215686917305, 0.3176470696926117, 0.3490196168422699, 0.3333333432674408, 0.3176470696926117, 0.29411765933036804, 0.2235294133424759, 0.11372549086809158, 0.0117647061124444, 0.239215686917305, 0.364705890417099, 0.45098039507865906, 0.45098039507865906, 0.4745098054409027, 0.4431372582912445, 0.41960784792900085, 0.30980393290519714, 0.23137255012989044, 0.13725490868091583, 0.05882352963089943, 0.05098039284348488, -0.06666667014360428, -0.12156862765550613, -0.06666667014360428, 0.06666667014360428, 0.11372549086809158, 0.13725490868091583, 0.20000000298023224, 0.27843138575553894, 0.3176470696926117, 0.3333333432674408, 0.30980393290519714, 0.3176470696926117, 0.32549020648002625, 0.18431372940540314, 0.13725490868091583, 0.05098039284348488, 0.1764705926179886, 0.26274511218070984, 0.3803921639919281, 0.46666666865348816, 0.4588235318660736, 0.4274509847164154, 0.41960784792900085, 0.3960784375667572, 0.3333333432674408, 0.18431372940540314, 0.1764705926179886, 0.07450980693101883, -0.12156862765550613, -0.08235294371843338, -0.027450980618596077, 0.04313725605607033, 0.12156862765550613, 0.16862745583057404, 0.20000000298023224, 0.26274511218070984, 0.30980393290519714, 0.3333333432674408, 0.35686275362968445, 0.32549020648002625, 0.2549019753932953, 0.21568627655506134, 0.15294118225574493, 0.1764705926179886, 0.2862745225429535, 0.20000000298023224, 0.35686275362968445, 0.3803921639919281, 0.41960784792900085, 0.4745098054409027, 0.45098039507865906, 0.5215686559677124, 0.4745098054409027, 0.3960784375667572, 0.3176470696926117, 0.239215686917305, 0.2078431397676468, 0.24705882370471954, 0.18431372940540314, 0.20000000298023224, 0.2235294133424759, 0.23137255012989044, 0.18431372940540314, 0.239215686917305, 0.3333333432674408, 0.3490196168422699, 0.30980393290519714, 0.23137255012989044, 0.14509804546833038, 0.1921568661928177, 0.15294118225574493, 0.2705882489681244, 0.2705882489681244, 0.11372549086809158, 0.35686275362968445, 0.364705890417099, 0.46666666865348816, 0.529411792755127, 0.5215686559677124, 0.5764706134796143, 0.46666666865348816, 0.43529412150382996, 0.2235294133424759, 0.21568627655506134, 0.24705882370471954, 0.1921568661928177, 0.16862745583057404, 0.16862745583057404, 0.15294118225574493, 0.10588235408067703, 0.12156862765550613, 0.18431372940540314, 0.2862745225429535, 0.2862745225429535, 0.3176470696926117, 0.3176470696926117, 0.23137255012989044, 0.1764705926179886, 0.2078431397676468, 0.2549019753932953, 0.30980393290519714, 0.24705882370471954, 0.3490196168422699, 0.3803921639919281, 0.45098039507865906, 0.5137255191802979, 0.46666666865348816, 0.4117647111415863, 0.32549020648002625, 0.37254902720451355, 0.2705882489681244, 0.29411765933036804, 0.1921568661928177, 0.13725490868091583, 0.16862745583057404, 0.18431372940540314, 0.12156862765550613, 0.06666667014360428, 0.09803921729326248, 0.1764705926179886, 0.24705882370471954, 0.2549019753932953, 0.30980393290519714, 0.30980393290519714, 0.23137255012989044, 0.23137255012989044, 0.24705882370471954, 0.3019607961177826, 0.30980393290519714, 0.3490196168422699, 0.3490196168422699, 0.3960784375667572, 0.34117648005485535, 0.41960784792900085, 0.35686275362968445, 0.3019607961177826, 0.3960784375667572, 0.41960784792900085, 0.3176470696926117, 0.20000000298023224, 0.13725490868091583, 0.1921568661928177, 0.2705882489681244, 0.2705882489681244, 0.15294118225574493, 0.15294118225574493, 0.06666667014360428, 0.14509804546833038, 0.2235294133424759, 0.16078431904315948, 0.24705882370471954, 0.20000000298023224, 0.2862745225429535, 0.26274511218070984, 0.21568627655506134, 0.239215686917305, 0.15294118225574493, 0.3176470696926117, 0.27843138575553894, 0.3803921639919281, 0.41960784792900085, 0.2862745225429535, 0.30980393290519714, 0.3019607961177826, 0.3960784375667572, 0.3803921639919281, 0.239215686917305, 0.2078431397676468, 0.08235294371843338, 0.12941177189350128, 0.239215686917305, 0.2235294133424759, 0.1921568661928177, 0.21568627655506134, 0.12156862765550613, 0.1921568661928177, 0.239215686917305, 0.2078431397676468, 0.2078431397676468, 0.09019608050584793, 0.15294118225574493, 0.239215686917305, 0.1764705926179886, 0.16862745583057404, 0.2235294133424759, 0.13725490868091583, 0.16862745583057404, 0.24705882370471954, 0.4117647111415863, 0.2549019753932953, 0.34117648005485535, 0.32549020648002625, 0.4431372582912445, 0.38823530077934265, 0.2078431397676468, 0.18431372940540314, 0.04313725605607033, 0.11372549086809158, 0.10588235408067703, 0.11372549086809158, 0.09803921729326248, 0.15294118225574493, 0.12156862765550613, 0.09019608050584793, 0.16862745583057404, 0.13725490868091583, 0.16078431904315948, 0.12941177189350128, 0.16078431904315948, 0.15294118225574493, 0.08235294371843338, 0.13725490868091583, 0.20000000298023224, 0.019607843831181526, 0.15294118225574493, 0.1921568661928177, 0.20000000298023224, 0.2078431397676468, 0.14509804546833038, 0.12941177189350128, 0.24705882370471954, 0.11372549086809158, 0.08235294371843338, 0.03529411926865578, -0.07450980693101883, 0.03529411926865578, -0.027450980618596077, 0.03529411926865578, 0.06666667014360428, 0.07450980693101883, 0.11372549086809158, 0.10588235408067703, -0.08235294371843338, 0.019607843831181526, 0.05882352963089943, 0.12156862765550613, 0.1764705926179886, 0.06666667014360428, 0.04313725605607033, 0.05098039284348488, 0.09019608050584793, 0.09803921729326248, 0.15294118225574493, 0.30980393290519714, 0.08235294371843338, 0.21568627655506134, 0.12156862765550613, 0.12156862765550613, 0.05882352963089943, -0.019607843831181526, 0.003921568859368563, -0.04313725605607033, -0.12156862765550613, -0.05098039284348488, -0.0117647061124444, 0.003921568859368563, 0.019607843831181526, 0.05882352963089943, 0.0117647061124444, -0.0117647061124444, -0.15294118225574493, -0.06666667014360428, -0.05098039284348488, 0.027450980618596077, 0.13725490868091583, 0.019607843831181526, -0.003921568859368563, 0.0117647061124444};
