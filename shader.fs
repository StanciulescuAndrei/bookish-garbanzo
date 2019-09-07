#version 330 core

in vec2 pixelCoords;

out vec4 fragmentColor;

uniform sampler2D temp;

void main()
{
	vec4 sample = texture(temp, pixelCoords);
	vec3 color;
	float value = sample.r / maxTemp;

	if(value < 0.2){
		color.b = 0.5 + 2.5 * value;
		color.r = 0;
		color.g = 0;
	}
	else if(value < 0.4){
		color.b = 1;
		color.r = 0;
		color.g = (value - 0.2) * 5;
	}
	else if(value < 0.6){
		color.b = 1.0 - (value - 0.4) * 5;
		color.r = (value - 0.4) * 5;
		color.g = 1;
	}
	else if(value < 0.8){
		color.b = 0;
		color.r = 1;
		color.g = 1.0 - (value - 0.6) * 5;
	}
	else{
		color.b = 0;
		color.g = 0;
		color.r = 1.0 - (value - 0.8) * 2.5;
	}


	fragmentColor = vec4(color, 1.0);
}