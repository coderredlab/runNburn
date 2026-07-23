#include <metal_stdlib>
using namespace metal;

// Darwin arm64 logf uses a 129-entry reciprocal/log center table and a cubic
// range-reduced polynomial in double, then rounds once to f32. Each double
// constant is split into a float-float pair so Metal follows that path without f64.
constant uint4 qwen_gdn_log_table[129] = {
    { 0x3f800000u, 0x00000000u, 0x00000000u, 0x00000000u },
    { 0x3f7e03f8u, 0x317e03f8u, 0x3bff0153u, 0x2f310679u },
    { 0x3f7c0fc1u, 0xb07c0fc0u, 0x3c7e0546u, 0xaff03fc2u },
    { 0x3f7a232du, 0xb15adec8u, 0x3cbdc8d8u, 0x2ffab623u },
    { 0x3f783e10u, 0xb2f83e10u, 0x3cfc14d8u, 0x30678330u },
    { 0x3f76603eu, 0xb2cfe134u, 0x3d1cf43eu, 0xb0402854u },
    { 0x3f74898du, 0x32bf0b76u, 0x3d3ba2c8u, 0xb09cd230u },
    { 0x3f72b9d6u, 0x32901e57u, 0x3d5a16ecu, 0xb0ee68e4u },
    { 0x3f70f0f1u, 0xb170f0f1u, 0x3d785186u, 0x2d0b1530u },
    { 0x3f6f2eb7u, 0x31fe21a2u, 0x3d8b29b7u, 0x316a37aeu },
    { 0x3f6d7304u, 0xb29467e2u, 0x3d9a0ebdu, 0xb11e42e3u },
    { 0x3f6bbdb3u, 0xb2b47d3du, 0x3da8d83au, 0xaf79e7c1u },
    { 0x3f6a0ea1u, 0xb1af8af9u, 0x3db78694u, 0x312e56b5u },
    { 0x3f6865acu, 0x32f6ec07u, 0x3dc61a2fu, 0xb11ce64eu },
    { 0x3f66c2b4u, 0x329039b1u, 0x3dd4936au, 0xb0b6a554u },
    { 0x3f652598u, 0x322bdc32u, 0x3de2f2a4u, 0x3175bc74u },
    { 0x3f638e39u, 0xb1e38e39u, 0x3df1383bu, 0x3162af2eu },
    { 0x3f61fc78u, 0x3161fc78u, 0x3dff648au, 0xb1624154u },
    { 0x3f607038u, 0x31e07038u, 0x3e06bbf4u, 0xb0cbdc6au },
    { 0x3f5ee95cu, 0x3299406fu, 0x3e0db957u, 0xb1ad0986u },
    { 0x3f5d67c9u, 0xb2b3e453u, 0x3e14aa98u, 0xb17c015cu },
    { 0x3f5beb62u, 0xb189731du, 0x3e1b8fe1u, 0x2e747ba0u },
    { 0x3f5a740eu, 0xb2b17e4bu, 0x3e22695bu, 0x31ccb7d2u },
    { 0x3f5901b2u, 0x305901b2u, 0x3e29372fu, 0x30e86d0eu },
    { 0x3f579436u, 0xb1d79436u, 0x3e2ff984u, 0xb1f586c3u },
    { 0x3f562b81u, 0xb22751fdu, 0x3e36b07fu, 0x31633a44u },
    { 0x3f54c77bu, 0x3054c77cu, 0x3e3d5c48u, 0x30843642u },
    { 0x3f53680du, 0x325a034eu, 0x3e43fd03u, 0x31241922u },
    { 0x3f520d21u, 0xb237cb7du, 0x3e4a92d5u, 0xb0c2ea53u },
    { 0x3f50b6a0u, 0xb250b6a0u, 0x3e511de1u, 0xae6a54e8u },
    { 0x3f4f6475u, 0xb2aefcc2u, 0x3e579e4au, 0x31e80bffu },
    { 0x3f4e168au, 0x32ee4a10u, 0x3e5e1434u, 0xb1bd2733u },
    { 0x3f4ccccdu, 0xb24ccccdu, 0x3e647fbeu, 0x31735344u },
    { 0x3f4b8728u, 0xb27e68f2u, 0x3e6ae10bu, 0x31b4fbb9u },
    { 0x3f4a4588u, 0xb1ca4588u, 0x3e71383bu, 0x31e2af2eu },
    { 0x3f4907dau, 0x329d0e23u, 0x3e77856eu, 0x31bdc593u },
    { 0x3f47ce0cu, 0x32f9c190u, 0x3e7dc8c3u, 0x31d5e3e3u },
    { 0x3f46980cu, 0x32d3018du, 0x3e82012du, 0xb234b2fcu },
    { 0x3f4565c8u, 0x32f6bf3bu, 0x3e851927u, 0x311ce439u },
    { 0x3f443730u, 0xb2f544fbu, 0x3e882c60u, 0xb1ca36a5u },
    { 0x3f430c31u, 0xb273cf3du, 0x3e8b3ae5u, 0x323aba61u },
    { 0x3f41e4bcu, 0xb229a824u, 0x3e8e44c6u, 0x30b4ccfeu },
    { 0x3f40c0c1u, 0xb27cfcfdu, 0x3e914a10u, 0xb18610d3u },
    { 0x3f3fa030u, 0xb1bfa030u, 0x3e944ad1u, 0xb2421796u },
    { 0x3f3e82fau, 0x313e82fau, 0x3e974716u, 0xb1a3dc5au },
    { 0x3f3d6910u, 0x328e0eccu, 0x3e9a3eedu, 0xb1acf055u },
    { 0x3f3c5264u, 0x313c5264u, 0x3e9d3263u, 0xb2296ba1u },
    { 0x3f3b3ee7u, 0x32069536u, 0x3ea02184u, 0x31d0d4fcu },
    { 0x3f3a2e8cu, 0xb2ba2e8cu, 0x3ea30c5eu, 0x310717b0u },
    { 0x3f392144u, 0xb0b92144u, 0x3ea5f2fdu, 0xb2288876u },
    { 0x3f381703u, 0xb1fd1fa4u, 0x3ea8d56cu, 0x31e5bf06u },
    { 0x3f370fbbu, 0x32b4337cu, 0x3eabb3b9u, 0xb20baa59u },
    { 0x3f360b61u, 0xb293e93fu, 0x3eae8deeu, 0xb027f636u },
    { 0x3f3509e7u, 0xb2eac8d7u, 0x3eb16418u, 0xb2546387u },
    { 0x3f340b41u, 0xb297e97fu, 0x3eb43641u, 0xb0b275a8u },
    { 0x3f330f63u, 0x32a51230u, 0x3eb70475u, 0x3222ba1eu },
    { 0x3f321643u, 0xb25e9bd4u, 0x3eb9cec0u, 0xb2144300u },
    { 0x3f311fd4u, 0xb28fe9dcu, 0x3ebc952bu, 0xaf0ae178u },
    { 0x3f302c0bu, 0x30302c0cu, 0x3ebf57c2u, 0xb18faa08u },
    { 0x3f2f3adeu, 0xb265fd43u, 0x3ec2168fu, 0xb1bc2e9du },
    { 0x3f2e4c41u, 0x32b93105u, 0x3ec4d19cu, 0x31d8284bu },
    { 0x3f2d602bu, 0x32b015acu, 0x3ec788f4u, 0x31e6cc59u },
    { 0x3f2c7692u, 0xb2f7ea71u, 0x3eca3ca1u, 0xb177ba41u },
    { 0x3f2b8f6au, 0xb1ebe532u, 0x3eccecacu, 0x308bf046u },
    { 0x3f2aaaabu, 0xb2aaaaabu, 0xbe934b11u, 0x326cb247u },
    { 0x3f29c84au, 0x328f40ffu, 0xbe90a22bu, 0xb250eb8du },
    { 0x3f28e83fu, 0x32ae2f81u, 0xbe8dfccbu, 0xb1569ae5u },
    { 0x3f280a81u, 0xb2afeaffu, 0xbe8b5ae6u, 0xb23acfb7u },
    { 0x3f272f05u, 0x3265e0a7u, 0xbe88bc74u, 0xb109f91fu },
    { 0x3f2655c4u, 0x3264b5eeu, 0xbe86216bu, 0xb1ec2c5cu },
    { 0x3f257eb5u, 0x30257eb6u, 0xbe8389c3u, 0xaf9ab0c4u },
    { 0x3f24a9cfu, 0x31ecb41au, 0xbe80f573u, 0x321d9397u },
    { 0x3f23d70au, 0x3275c28fu, 0xbe7cc8e3u, 0xb1cb3b38u },
    { 0x3f23065eu, 0x327eb9f3u, 0xbe77ad6fu, 0xb11b9ffdu },
    { 0x3f2237c3u, 0x322c5b3fu, 0xbe729878u, 0x2e477f70u },
    { 0x3f216b31u, 0x323aa3f1u, 0xbe6d89eeu, 0x31f2b76cu },
    { 0x3f20a0a1u, 0xb2bebebfu, 0xbe6881c0u, 0x31d9aa18u },
    { 0x3f1fd80au, 0xb01fd80au, 0xbe637fdeu, 0xb15e01eeu },
    { 0x3f1f1166u, 0xb1c6d5c0u, 0xbe5e843au, 0x317884eau },
    { 0x3f1e4cadu, 0x320f757du, 0xbe598ec3u, 0x318a431cu },
    { 0x3f1d89d9u, 0xb2c4ec4fu, 0xbe549f6au, 0x30dd4987u },
    { 0x3f1cc8e1u, 0x32c187f6u, 0xbe4fb620u, 0xb16112ccu },
    { 0x3f1c09c1u, 0xb2c7ec7fu, 0xbe4ad2d7u, 0x30c23fa0u },
    { 0x3f1b4c70u, 0xb2c21f8cu, 0xbe45f57fu, 0xb1b38fe9u },
    { 0x3f1a90e8u, 0xb21a90e8u, 0xbe411e0bu, 0xb12a3478u },
    { 0x3f19d723u, 0xb215086au, 0xbe3c4c6cu, 0xb128898eu },
    { 0x3f191f1au, 0x32a2b10cu, 0xbe378094u, 0xb1b756acu },
    { 0x3f1868c8u, 0x311868c8u, 0xbe32ba76u, 0x3039f663u },
    { 0x3f17b426u, 0xb197b426u, 0xbe2dfa03u, 0xb1b543dbu },
    { 0x3f17012eu, 0x3017012eu, 0xbe293f2fu, 0xb11436b2u },
    { 0x3f164fdau, 0x32d812cau, 0xbe2489ecu, 0xb0cced58u },
    { 0x3f15a025u, 0x32d012b4u, 0xbe1fda2du, 0xb133251au },
    { 0x3f14f209u, 0x329e412au, 0xbe1b2fe6u, 0x31fea6ffu },
    { 0x3f144581u, 0xb2d774ffu, 0xbe168b08u, 0xb1c86814u },
    { 0x3f139a86u, 0xb26fdb19u, 0xbe11eb89u, 0xb1a49c20u },
    { 0x3f12f114u, 0xb2f7f6d1u, 0xbe0d515cu, 0x306e046bu },
    { 0x3f124925u, 0xb2db6db7u, 0xbe08bc74u, 0xb089f91fu },
    { 0x3f11a2b4u, 0xb26ca864u, 0xbe042cc6u, 0x31a61c60u },
    { 0x3f10fdbcu, 0x3110fdbcu, 0xbdff448au, 0x30c48553u },
    { 0x3f105a38u, 0x32c67c0eu, 0xbdf639ccu, 0xb0428448u },
    { 0x3f0fb824u, 0xb18fb824u, 0xbded393bu, 0xb06111a8u },
    { 0x3f0f177au, 0xb21808f2u, 0xbde442c0u, 0xafde2592u },
    { 0x3f0e7835u, 0x32da2812u, 0xbddb5644u, 0xb15ad5b2u },
    { 0x3f0dda52u, 0x300dda52u, 0xbdd273b2u, 0xaf31bc38u },
    { 0x3f0d3dcbu, 0x310d3dcbu, 0xbdc99af3u, 0x3029ad9eu },
    { 0x3f0ca29cu, 0x308ca29cu, 0xbdc0cbf1u, 0xb1740e3fu },
    { 0x3f0c08c1u, 0xb2e7ee7fu, 0xbdb80698u, 0xb12eac0eu },
    { 0x3f0b7034u, 0x32942738u, 0xbdaf4ad2u, 0xb159791du },
    { 0x3f0ad8f3u, 0xb08ad8f3u, 0xbda6988bu, 0x3037e055u },
    { 0x3f0a42f8u, 0x32e0acd4u, 0xbd9defadu, 0xb0fa3dccu },
    { 0x3f09ae41u, 0xb2eca37fu, 0xbd955025u, 0xb088e2f5u },
    { 0x3f091ac7u, 0x326ba606u, 0xbd8cb9dfu, 0x316b9aaau },
    { 0x3f088889u, 0xb2eeeeefu, 0xbd842cc6u, 0x31261c60u },
    { 0x3f07f781u, 0xb2f010ffu, 0xbd77518eu, 0xac570f80u },
    { 0x3f0767abu, 0x32be69c9u, 0xbd665b9eu, 0xb0dddb2du },
    { 0x3f06d905u, 0x3288f469u, 0xbd557797u, 0x30f04ef0u },
    { 0x3f064b8au, 0x32fbcda4u, 0xbd44a551u, 0x30b604ccu },
    { 0x3f05bf37u, 0x32c259dcu, 0xbd33e4a8u, 0x30d2b44au },
    { 0x3f053408u, 0x32a6810au, 0xbd233577u, 0x30bd21c1u },
    { 0x3f04a9fau, 0xb25fded6u, 0xbd129799u, 0xb0f8d184u },
    { 0x3f042108u, 0x32842108u, 0xbd020aecu, 0xb09e7444u },
    { 0x3f039930u, 0x32a47f7cu, 0xbce31e97u, 0xb0414aafu },
    { 0x3f03126fu, 0xb2d0e560u, 0xbcc24929u, 0xb00c8cacu },
    { 0x3f028cc0u, 0xb2828cc0u, 0xbca19549u, 0xafb30198u },
    { 0x3f020821u, 0xb2fbefbfu, 0xbc8102b3u, 0x2fed94f7u },
    { 0x3f01848eu, 0xb2ae0a1eu, 0xbc412245u, 0xaee228aau },
    { 0x3f010204u, 0x31010204u, 0xbc0080acu, 0x2fa77219u },
    { 0x3f008081u, 0xb2fefeffu, 0xbb80402bu, 0x2ed4ee99u },
    { 0x3f000000u, 0x00000000u, 0x00000000u, 0x00000000u },
};

inline float qwen_gdn_log_f32(float value)
{
    uint word = as_type<uint>(value);
    uint magnitude = word & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return value + value;
    if (magnitude == 0u) return -1.0f / 0.0f;
    if ((word >> 31u) != 0u) return as_type<float>(0x7fc00000u);
    if (magnitude == 0x7f800000u) return value;
    if (magnitude < 0x00800000u) {
        value *= 0x1p23f;
        word = as_type<uint>(value) - (23u << 23u);
    }

    uint table_index = ((word & 0x007fffffu) + 0x00008000u) >> 16u;
    int exponent = as_type<int>(word - 0x3f3f8000u) >> 23;
    float mantissa = as_type<float>((word & 0x007fffffu) | 0x3f800000u);
    uint4 table_word = qwen_gdn_log_table[table_index];
    qwen_moe_float2 invc = {
        as_type<float>(table_word.x),
        as_type<float>(table_word.y)
    };
    qwen_moe_float2 logc = {
        as_type<float>(table_word.z),
        as_type<float>(table_word.w)
    };
    qwen_moe_float2 reduced = qwen_moe_float2_add(
        qwen_moe_float2_multiply({ mantissa, 0.0f }, invc),
        { -1.0f, 0.0f });
    qwen_moe_float2 reduced2 = qwen_moe_float2_multiply(reduced, reduced);
    const qwen_moe_float2 cubic = {
        as_type<float>(0x3eaaab0bu),
        as_type<float>(0xb22a3345u)
    };
    const qwen_moe_float2 quadratic = {
        as_type<float>(0xbf000030u),
        as_type<float>(0xadd27480u)
    };
    const qwen_moe_float2 linear = {
        as_type<float>(0x3f800000u),
        as_type<float>(0xac999a00u)
    };
    const qwen_moe_float2 ln2 = {
        as_type<float>(0x3f317218u),
        as_type<float>(0xb102e308u)
    };
    qwen_moe_float2 polynomial = qwen_moe_float2_add(
        quadratic,
        qwen_moe_float2_multiply(cubic, reduced));
    qwen_moe_float2 result = qwen_moe_float2_add(
        logc,
        qwen_moe_float2_multiply(ln2, { (float)exponent, 0.0f }));
    result = qwen_moe_float2_add(
        result,
        qwen_moe_float2_multiply(linear, reduced));
    result = qwen_moe_float2_add(
        result,
        qwen_moe_float2_multiply(reduced2, polynomial));
    volatile float rounded = result.hi + result.lo;
    return rounded;
}


// GDN alpha/beta 전처리 (token, head 별, in-place):
//   beta[t,h]  = sigmoid(beta[t,h])
//   alpha[t,h] = softplus(alpha[t,h] + dt_bias[h]) * ssm_a[h]
// 높이가 1인 decode dispatch와 여러 token을 한 번에 처리하는 prefill dispatch를
// 같은 커널로 처리한다.
kernel void gdn_alpha_beta(
    device float*       alpha     [[buffer(0)]],
    device float*       beta      [[buffer(1)]],
    device const float* dt_bias   [[buffer(2)]],
    device const float* ssm_a     [[buffer(3)]],
    constant uint&      num_heads [[buffer(4)]],
    uint2 position [[thread_position_in_grid]])
{
    uint h = position.x;
    if (h >= num_heads) return;
    uint index = position.y * num_heads + h;
    beta[index] = 1.0f / (1.0f + qwen_moe_exp_f32(-beta[index]));
    float a_biased = alpha[index] + dt_bias[h];
    float sp = qwen_gdn_log_f32(1.0f + qwen_moe_exp_f32(a_biased));
    alpha[index] = sp * ssm_a[h];
}

