$input a_position, a_texcoord0
$output v_texcoord0, v_position

#include "../common/common.sh"

SAMPLER2D(s_heightTexture, 3);

void main()
{
    vec3 pos = a_position;

    // 从高度纹理采样
    float height = texture2D(s_heightTexture, a_texcoord0).x * 255.0;
    pos.y = height;

    v_texcoord0 = a_texcoord0;
    v_position = pos;

    gl_Position = mul(u_modelViewProj, vec4(pos, 1.0));
}
