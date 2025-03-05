$input v_texcoord0, v_position

#include "../common/common.sh"

// 纹理采样器
SAMPLER2D(s_grassTexture, 0);
SAMPLER2D(s_rockTexture, 1);
SAMPLER2D(s_snowTexture, 2);

// 参数
uniform vec4 u_vegetationParams;

void main()
{
    // 将高度归一化到 0-1 范围 (假设最大高度为 255)
    float heightFactor = v_position.y / 255.0;

    // 采样纹理
    vec4 grassColor = texture2D(s_grassTexture, v_texcoord0);
    vec4 rockColor = texture2D(s_rockTexture, v_texcoord0);
    vec4 snowColor = texture2D(s_snowTexture, v_texcoord0);

    // 基于高度参数进行混合
    float grassThreshold = u_vegetationParams.x;
    float rockThreshold = u_vegetationParams.y;
    float snowThreshold = u_vegetationParams.z;
    float blendSharpness = u_vegetationParams.w;

    // 计算混合权重
    float grassWeight = 1.0 - smoothstep(grassThreshold - blendSharpness, grassThreshold + blendSharpness, heightFactor);
    float rockWeight = smoothstep(grassThreshold - blendSharpness, grassThreshold + blendSharpness, heightFactor) *
                      (1.0 - smoothstep(rockThreshold - blendSharpness, rockThreshold + blendSharpness, heightFactor));
    float snowWeight = smoothstep(rockThreshold - blendSharpness, rockThreshold + blendSharpness, heightFactor);

    // 归一化权重
    float totalWeight = grassWeight + rockWeight + snowWeight;
    grassWeight /= totalWeight;
    rockWeight /= totalWeight;
    snowWeight /= totalWeight;

    // 混合纹理
    vec4 finalColor = grassColor * grassWeight + rockColor * rockWeight + snowColor * snowWeight;

    // 增加基于高度的阴影效果，增加深度感
    float shadeFactor = 0.7 + 0.3 * heightFactor;

    gl_FragColor = finalColor * shadeFactor;
}
