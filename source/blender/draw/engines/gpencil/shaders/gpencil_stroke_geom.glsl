uniform mat4 ModelViewProjectionMatrix;
uniform vec2 Viewport;

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 7) out;

in vec4 finalColor[4];
in float finalThickness[4];

out vec4 mColor;
out vec2 mTexCoord;

/* project 3d point to 2d on screen space */
vec2 toScreenSpace(vec4 vertex)
{
	return vec2(vertex.xy / vertex.w) * Viewport;
}

void main(void)
{
	float MiterLimit = 0.75;

	/* receive 4 points */
	vec4 P0 = gl_in[0].gl_Position;
	vec4 P1 = gl_in[1].gl_Position;
	vec4 P2 = gl_in[2].gl_Position;
	vec4 P3 = gl_in[3].gl_Position;

	/* get the four vertices passed to the shader */
	vec2 sp0 = toScreenSpace(P0);	// start of previous segment
	vec2 sp1 = toScreenSpace(P1);	// end of previous segment, start of current segment
	vec2 sp2 = toScreenSpace(P2);	// end of current segment, start of next segment
	vec2 sp3 = toScreenSpace(P3);   // end of next segment
	
	/* culling outside viewport */
	vec2 area = Viewport * 4.0;
	if (sp1.x < -area.x || sp1.x > area.x) return;
	if (sp1.y < -area.y || sp1.y > area.y) return;
	if (sp2.x < -area.x || sp2.x > area.x) return;
	if (sp2.y < -area.y || sp2.y > area.y) return;

	/* determine the direction of each of the 3 segments (previous, current, next) */
	vec2 v0 = normalize(sp1 - sp0);
	vec2 v1 = normalize(sp2 - sp1);
	vec2 v2 = normalize(sp3 - sp2);

	/* determine the normal of each of the 3 segments (previous, current, next) */
	vec2 n0 = vec2(-v0.y, v0.x);
	vec2 n1 = vec2(-v1.y, v1.x);
	vec2 n2 = vec2(-v2.y, v2.x);

	/* determine miter lines by averaging the normals of the 2 segments */
	vec2 miter_a = normalize(n0 + n1);	// miter at start of current segment
	vec2 miter_b = normalize(n1 + n2);  // miter at end of current segment

    /* determine the length of the miter by projecting it onto normal and then inverse it */
	float an1 = dot(miter_a, n1);
	float bn1 = dot(miter_b, n2);
	if (an1 == 0) an1 = 1;
	if (bn1 == 0) bn1 = 1;
	float length_a = finalThickness[1] / an1;
	float length_b = finalThickness[2] / bn1;
	if (length_a <= 0.0) length_a = 0.01;
	if (length_b <= 0.0) length_b = 0.01;

	/* prevent excessively long miters at sharp corners */
	if (dot(v0, v1) < -MiterLimit) {
		miter_a = n1;
		length_a = finalThickness[1];

		/* close the gap */
		if (dot(v0, n1) > 0) {
			mTexCoord = vec2(0, 0);
			mColor = finalColor[1];
			gl_Position = vec4((sp1 + finalThickness[1] * n0) / Viewport, 0.0, 1.0);
			EmitVertex();

			mTexCoord = vec2(0, 0);
			mColor = finalColor[1];
			gl_Position = vec4((sp1 + finalThickness[1] * n1) / Viewport, 0.0, 1.0);
			EmitVertex();

			mTexCoord = vec2(0, 0.5);
			mColor = finalColor[1];
			gl_Position = vec4(sp1 / Viewport, 0.0, 1.0);
			EmitVertex();

			EndPrimitive();
		}
		else {
			mTexCoord = vec2(0, 1);
			mColor = finalColor[1];
			gl_Position = vec4((sp1 - finalThickness[1] * n1) / Viewport, 0.0, 1.0);
			EmitVertex();

			mTexCoord = vec2(0, 1);
			mColor = finalColor[1];
			gl_Position = vec4((sp1 - finalThickness[1] * n0) / Viewport, 0.0, 1.0);
			EmitVertex();

			mTexCoord = vec2(0, 0.5);
			mColor = finalColor[1];
			gl_Position = vec4(sp1 / Viewport, 0.0, 1.0);
			EmitVertex();

			EndPrimitive();
		}
	}

	if (dot(v1, v2) < -MiterLimit) {
		miter_b = n1;
		length_b = finalThickness[2];
	}

	/* generate the triangle strip */
	mTexCoord = vec2(0, 0);
	mColor = finalColor[1];
	gl_Position = vec4((sp1 + length_a * miter_a) / Viewport, 0.0, 1.0);
	EmitVertex();

	mTexCoord = vec2(0, 1);
	mColor = finalColor[1];
	gl_Position = vec4((sp1 - length_a * miter_a) / Viewport, 0.0, 1.0);
	EmitVertex();

	mTexCoord = vec2(0, 0);
	mColor = finalColor[2];
	gl_Position = vec4((sp2 + length_b * miter_b) / Viewport, 0.0, 1.0);
	EmitVertex();

	mTexCoord = vec2(0, 1);
	mColor = finalColor[2];
	gl_Position = vec4((sp2 - length_b * miter_b) / Viewport, 0.0, 1.0);
	EmitVertex();

	EndPrimitive();
}