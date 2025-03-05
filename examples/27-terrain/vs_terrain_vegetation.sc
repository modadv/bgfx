$input a_position, a_texcoord0
$output v_texcoord0, v_position

#include "../common/common.sh"

void main()
{
    vec3 pos = a_position;

    v_texcoord0 = a_texcoord0;
    v_position = pos;

    gl_Position = mul(u_modelViewProj, vec4(pos, 1.0));
}
