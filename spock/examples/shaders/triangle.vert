// Headless-triangle vertex shader.
//
// No vertex buffer needed: gl_VertexIndex picks one of three hardcoded
// positions and matching colors. The fragment shader receives the color
// via a varying.

#version 450

layout(location = 0) out vec3 frag_color;

void main() {
    // Three positions covering most of the [-1, 1] x [-1, 1] viewport.
    vec2 positions[3] = vec2[3](
        vec2( 0.0, -0.7),
        vec2( 0.7,  0.7),
        vec2(-0.7,  0.7)
    );
    // Red, green, blue corners.
    vec3 colors[3] = vec3[3](
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0)
    );
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    frag_color = colors[gl_VertexIndex];
}
