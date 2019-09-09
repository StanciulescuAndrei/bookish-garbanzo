#version 330 core

in vec2 pixelCoords;
out vec4 fragmentColor;

uniform sampler2D smplr;

uniform vec2 tempRange;

void main()
{
	vec4 sample = texture(smplr, pixelCoords);
	vec3 color;
	
	//float minTemp = tempRange.x;
	//float maxTemp = tempRange.y + tempRange.y / 10;
	float minTemp = 0;
	float maxTemp = 5000;
	float value = (sample.r - minTemp) / (maxTemp - minTemp);

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
	else if(value <= 1){
		color.b = 0;
		color.g = 0;
		color.r = 1.0 - (value - 0.8) * 2.5;
	}
	else{
		color.r = 0.5;
		color.b = 0.0;
		color.g = 0.0;
	}


	fragmentColor = vec4(color, 1.0);
}