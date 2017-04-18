// ImGui - standalone example application for Glfw + OpenGL 3, using programmable pipeline
// If you are new to ImGui, see examples/README.txt and documentation at the top of imgui.cpp.

#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"
#include <stdio.h>
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <vector>
#include "micro_gps.h"

#define IM_ARRAYSIZE(_ARR)  ((int)(sizeof(_ARR)/sizeof(*_ARR)))

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error %d: %s\n", error, description);
}


GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path){

	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if(VertexShaderStream.is_open()){
		std::string Line = "";
		while(getline(VertexShaderStream, Line))
			VertexShaderCode += "\n" + Line;
		VertexShaderStream.close();
	}else{
		printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
		getchar();
		return 0;
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
	if(FragmentShaderStream.is_open()){
		std::string Line = "";
		while(getline(FragmentShaderStream, Line))
			FragmentShaderCode += "\n" + Line;
		FragmentShaderStream.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;


	// Compile Vertex Shader
	printf("Compiling shader : %s\n", vertex_file_path);
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}



	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragment_file_path);
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}



	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> ProgramErrorMessage(InfoLogLength+1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}


	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}


GLuint loadBMP_custom(const char * imagepath) {
  // Data read from the header of the BMP file
  unsigned char header[54]; // Each BMP file begins by a 54-bytes header
  unsigned int dataPos;     // Position in the file where the actual data begins
  unsigned int width, height;
  unsigned int imageSize;   // = width*height*3
  // Actual RGB data
  unsigned char * data;

  // Open the file
  FILE * file = fopen(imagepath,"rb");
  if (!file){printf("Image could not be opened\n"); return 0;}

  if ( fread(header, 1, 54, file)!=54 ){ // If not 54 bytes read : problem
    printf("Not a correct BMP file\n");
    return false;
  }

  if ( header[0]!='B' || header[1]!='M' ){
      printf("Not a correct BMP file\n");
      return 0;
  }

  // Read ints from the byte array
  dataPos    = *(int*)&(header[0x0A]);
  imageSize  = *(int*)&(header[0x22]);
  width      = *(int*)&(header[0x12]);
  height     = *(int*)&(header[0x16]);

  // Some BMP files are misformatted, guess missing information
  if (imageSize==0)    imageSize=width*height*3; // 3 : one byte for each Red, Green and Blue component
  if (dataPos==0)      dataPos=54; // The BMP header is done that way

  // Create a buffer
  data = new unsigned char [imageSize];

  // Read the actual data from the file into the buffer
  fread(data,1,imageSize,file);

  //Everything is in memory now, the file can be closed
  fclose(file);

  // Create one OpenGL texture
  GLuint textureID;
  glGenTextures(1, &textureID);

  // "Bind" the newly created texture : all future texture functions will modify this texture
  glBindTexture(GL_TEXTURE_2D, textureID);

  // Give the image to OpenGL
  glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);

  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);

 return textureID;

}



int main(int, char**)
{
    // Setup window
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(1280, 720, "ImGui OpenGL3 example", NULL, NULL);
    glfwMakeContextCurrent(window);
    gl3wInit();



    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);
    // An array of 3 vectors which represents 3 vertices
    static const GLfloat g_vertex_buffer_data[] = {
       -1.0f, 1.0f, 0.0f,
       -1.0f, -1.0f, 0.0f,
       1.0f,  1.0f, 0.0f,
       1.0f, -1.0f, 0.0f,
       1.0f, 1.0f, 0.0f,
       -1.0f, -1.0f, 0.0f
    };


    static const GLfloat g_uv_buffer_data[] = {
      0.0f, 1.0f,
      0.0f, 0.0f,
      1.0f, 1.0f,
      1.0f, 0.0f,
      1.0f, 1.0f,
      0.0f, 0.0f
    };

    // This will identify our vertex buffer
    GLuint vertexbuffer;
    // Generate 1 buffer, put the resulting identifier in vertexbuffer
    glGenBuffers(1, &vertexbuffer);
    // The following commands will talk about our 'vertexbuffer' buffer
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    // Give our vertices to OpenGL.
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);



    GLuint uvbuffer;
  	glGenBuffers(1, &uvbuffer);
  	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
  	glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);



    GLuint programID = LoadShaders( "test.vert", "test.frag" );

    GLuint TextureID  = glGetUniformLocation(programID, "myTextureSampler");
    GLuint Texture = loadBMP_custom("lena.bmp");



    // Setup ImGui binding
    ImGui_ImplGlfwGL3_Init(window, true);

    // Load Fonts
    // (there is a default font, this is only if you want to change it. see extra_fonts/README.txt for more details)
    //ImGuiIO& io = ImGui::GetIO();
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("../../extra_fonts/Cousine-Regular.ttf", 15.0f);
    //io.Fonts->AddFontFromFileTTF("../../extra_fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../extra_fonts/ProggyClean.ttf", 13.0f);
    //io.Fonts->AddFontFromFileTTF("../../extra_fonts/ProggyTiny.ttf", 10.0f);
    //io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());

    bool show_test_window = false;
    bool show_another_window = false;
    ImVec4 clear_color = ImColor(114, 144, 154);

    static char map_path[] = "enter map path";


    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);


        // ImGui::SetNextWindowSize(ImVec2(display_w/2, 1));
        // ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        // ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
        // ImGui::Begin("Menu", &show_another_window, ImGuiWindowFlags_NoTitleBar|
        //                                                       ImGuiWindowFlags_NoResize|
        //                                                       ImGuiWindowFlags_NoMove|
        //                                                       ImGuiWindowFlags_MenuBar|
        //                                                       ImGuiWindowFlags_NoScrollbar);
        // ImGui::End();
        // ImGui::PopStyleVar();


        // printf("%d x %d\n", display_w, display_h);
        // 1. Show a simple window
        // Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appears in a window automatically called "Debug"
        {

            ImGui::Begin("First Window");
            static float f = 0.0f;
            ImGui::Text("Hello, world!");
            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
            ImGui::ColorEdit3("clear color", (float*)&clear_color);
            if (ImGui::Button("Test Window")) show_test_window ^= 1;
            if (ImGui::Button("Another Window")) show_another_window ^= 1;
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        ImGui::SetNextWindowSize(ImVec2(350,display_h/2));
        ImGui::SetNextWindowPos(ImVec2(display_w/2-350, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
        ImGui::Begin("Settings", &show_another_window, ImGuiWindowFlags_NoTitleBar|
                                                              ImGuiWindowFlags_NoResize|
                                                              ImGuiWindowFlags_NoMove|
                                                              ImGuiWindowFlags_NoScrollbar);


        if (ImGui::CollapsingHeader("Dataset", ImGuiTreeNodeFlags_Leaf)) {
          const char* items[] = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ", "KKKK" };
          static int item2 = -1;
          ImGui::Combo("training", &item2, items, IM_ARRAYSIZE(items));   // Combo using proper array. You can also pass a callback to retrieve array value, no need to create/copy an array just for that.
          static int frame_idx = 0;
          ImGui::SliderInt("testing index", &frame_idx, 0, 1000);
        }

        if (ImGui::CollapsingHeader("Testing", ImGuiTreeNodeFlags_Leaf)) {
          // static int num_dims = 8;
          static float cell_size = 50.0f;
          // ImGui::SliderInt("# dimensions", &num_dims, 2, 64);
          ImGui::SliderFloat("cell size", &cell_size, 10.0f, 100.0f);

          const char* items[] = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ", "KKKK" };
          static int item2 = -1;
          ImGui::Combo("database", &item2, items, IM_ARRAYSIZE(items));   // Combo using proper array. You can also pass a callback to retrieve array value, no need to create/copy an array just for that.

          const char* items3[] = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ", "KKKK" };
          static int item4 = -1;
          ImGui::Combo("PCA basis", &item4, items3, IM_ARRAYSIZE(items3));   // Combo using proper array. You can also pass a callback to retrieve array value, no need to create/copy an array just for that.
          if (ImGui::Button("test", ImVec2(-1, 0))) {
            printf("Clicked\n");
          }
        }

        if (ImGui::CollapsingHeader("Training", ImGuiTreeNodeFlags_Leaf)) {
          static int num_feature_per_image = 50;
          static int num_dims_training = 8;
          ImGui::SliderInt("sample size", &num_feature_per_image, 20, 100);
          ImGui::SliderInt("# dimensions", &num_dims_training, 2, 64);

          static char map_path[] = "enter map path";
          ImGui::InputText("###map_path", map_path, sizeof(map_path));
          ImGui::SameLine();
          if (ImGui::Button("generate", ImVec2(-1, 0))) {
            printf("Clicked\n");
          }

          ImGui::BeginGroup();
          static char database_path[] = "enter database path";
          ImGui::InputText("###database_path", database_path, sizeof(database_path));

          // ImGui::PushItemWidth(100);
          // if (ImGui::Button("process")) {
          //   printf("Clicked\n");
          // }
          // ImGui::PopItemWidth();

          static char basis_path[] = "enter basis path";
          ImGui::InputText("###basis_path", basis_path, sizeof(basis_path));
          ImGui::EndGroup();
          ImVec2 size = ImGui::GetItemRectSize();
          ImGui::SameLine();
          ImGui::Button("process", ImVec2(-1, size.y));


          // ImGui::SameLine();
          // ImGui::PushItemWidth(100);
          // if (ImGui::Button("generate")) {
          //   printf("Clicked\n");
          // }
          // ImGui::PopItemWidth();
        }
        if (ImGui::CollapsingHeader("Monitor", ImGuiTreeNodeFlags_Leaf)) {
          ImVec2 size = ImGui::GetItemRectSize();
          ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
          const float values[10] = { 0.5f, 0.20f, 0.80f, 0.60f, 0.25f, 0.5f, 0.20f, 0.80f, 0.60f, 0.25f};
          ImGui::PlotHistogram("##values", values, IM_ARRAYSIZE(values), 0, NULL, 0.0f, 1.0f, ImVec2(size.x, 100));


        }

        ImGui::End();
        ImGui::PopStyleVar();


        ImGui::SetNextWindowSize(ImVec2(200,100));
        ImGui::SetNextWindowPos(ImVec2(200,100));
        ImGui::Begin("Another Window2", &show_another_window);
        ImGui::Text("Yoyo");
        ImGui::End();


        // 3. Show the ImGui test window. Most of the sample code is in ImGui::ShowTestWindow()
        if (show_test_window)
        {
            ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
            ImGui::ShowTestWindow(&show_test_window);
        }

        // Rendering
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(programID);
        glViewport(640, 360, 512, 512);


        glActiveTexture(GL_TEXTURE0);
    		glBindTexture(GL_TEXTURE_2D, Texture);
    		// Set our "myTextureSampler" sampler to user Texture Unit 0
    		glUniform1i(TextureID, 0);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
           0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
           3,                  // size
           GL_FLOAT,           // type
           GL_FALSE,           // normalized?
           0,                  // stride
           (void*)0            // array buffer offset
        );

        glEnableVertexAttribArray(1);
    		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
    		glVertexAttribPointer(
    			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
    			2,                                // size : U+V => 2
    			GL_FLOAT,                         // type
    			GL_FALSE,                         // normalized?
    			0,                                // stride
    			(void*)0                          // array buffer offset
    		);


        glActiveTexture(GL_TEXTURE0);
    		glBindTexture(GL_TEXTURE_2D, Texture);
    		// Set our "myTextureSampler" sampler to user Texture Unit 0
    		glUniform1i(TextureID, 0);

        // Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, 6); // Starting from vertex 0; 3 vertices total -> 1 triangle

        glDisableVertexAttribArray(0);
    		glDisableVertexAttribArray(1);

        ImGui::Render();


        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplGlfwGL3_Shutdown();
    glfwTerminate();

    return 0;
}
