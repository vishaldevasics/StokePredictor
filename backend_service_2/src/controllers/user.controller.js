import { asyncHandler } from "../utils/asyncHandler.js";
import { ApiError } from "../utils/ApiError.js";
import { User } from "../models/user.model.js";
import { uploadOnCloudinary } from "../utils/cloudinary.js";
import { ApiResponse } from "../utils/ApiResponse.js";
import jwt from "jsonwebtoken";

const generateAccessAndRefreshTokens = async (userId) => {
  try {
    const user = await User.findById(userId);
    const accessToken = user.generateAccessToken();
    const refreshToken = user.generateRefreshToken();

    user.refreshToken = refreshToken;
    await user.save({ validateBeforeSave: false });

    return { accessToken, refreshToken };
  } catch (error) {
    throw new ApiError(500, "Something went wrong while generating tokens");
  }
};

const registerUser = asyncHandler(async (req, res) => {
  const { email, fname, lname, password } = req.body;

  if ([email, fname, lname, password].some((field) => field?.trim() === "")) {
    throw new ApiError(400, "All fields are required.");
  }

  const existedUser = await User.findOne({
    $or: [{ email }],
  });

  if (existedUser) {
    throw new ApiError(409, "User with email already exists.");
  }

  const user = await User.create({
    email,
    password,
    fname,
    lname
  });
  const createdUser = await User.findById(user._id).select("-password -refreshToken");

  if (!createdUser) {
    throw new ApiError(500, "Something went wrong while registering the user");
  }
  return res.status(201).json(
    new ApiResponse(200, createdUser, "User registered successfully")
  );
});


const loginUser = asyncHandler(async (req, res) => {
  const { email, password } = req.body;

  if (!email) {
    throw new ApiError(400, "Email is required");
  }

  const user = await User.findOne({
    $or: [{ email }],
  });

  if (!user) {
    throw new ApiError(404, "User does not exist");
  }

  const isPasswordValid = await user.isPasswordCorrect(password);

  if (!isPasswordValid) {
    throw new ApiError(401, "Invalid user credentials");
  }

  const { accessToken, refreshToken } = await generateAccessAndRefreshTokens(user._id);

  const loggedInUser = await User.findById(user._id).select("-password -refreshToken");

  // Set the cookies with options
  const options = {
    httpOnly: true, // Only accessible by the server
    secure: process.env.NODE_ENV === "production", // Only use HTTPS in production
    sameSite: "None", // Allow cross-origin cookies
  };

  return res
    .status(200)
    .cookie("accessToken", accessToken, options)
    .cookie("refreshToken", refreshToken, options)
    .json(
      new ApiResponse(
        200,
        {
          user: loggedInUser,
          accessToken,
          refreshToken,
        },
        "User logged in successfully"
      )
    );
});

const logoutUser = asyncHandler(async (req, res) => {
  await User.findByIdAndUpdate(
    req.user._id,
    {
      $set: {
        refreshToken: undefined,
      },
    },
    {
      new: true,
    }
  );

  const options = {
    httpOnly: true,
    secure: true,
  };

  return res
    .status(200)
    .clearCookie("accessToken", options)
    .clearCookie("refreshToken", options)
    .json(new ApiResponse(200, {}, "User logged out"));
});

const refreshAccessToken = asyncHandler(async (req, res) => {
  const incomingRefreshToken = req.cookies.refreshToken || req.body.refreshToken;

  if (!incomingRefreshToken) {
    throw new ApiError(401, "Unauthorized Request");
  }

  try {
    const decodedToken = jwt.verify(
      incomingRefreshToken,
      process.env.REFRESH_TOKEN_SECRET
    );

    const user = await User.findById(decodedToken?._id);

    if (!user) {
      throw new ApiError(401, "Invalid refresh token");
    }

    if (incomingRefreshToken !== user?.refreshToken) {
      throw new ApiError(401, "Refresh token is expired or used");
    }

    const options = {
      httpOnly: true,
      secure: true,
    };

    const { accessToken, refreshToken: newRefreshToken } = await generateAccessAndRefreshTokens(user._id);

    return res
      .status(200)
      .cookie("accessToken", accessToken, options)
      .cookie("refreshToken", newRefreshToken, options)
      .json(
        new ApiResponse(
          200,
          {
            accessToken,
            refreshToken: newRefreshToken,
          },
          "Access token refreshed"
        )
      );
  } catch (error) {
    throw new ApiError(401, error?.message || "Invalid refresh token");
  }
});

const getUserData = asyncHandler(async (req, res) => {
  const user = await User.findById(req.user._id).select("-password -refreshToken");
  return res.status(200).json(new ApiResponse(200, user, "User data fetched successfully"));
});

const submitAssignment = asyncHandler(async (req, res) => {
  const { courseName, subjectName, assignmentName } = req.body;

  if (!courseName || !subjectName || !assignmentName) {
    throw new ApiError(400, "Course name, subject name, and assignment name are required.");
  }

  // Check if there's an uploaded file and get its Cloudinary URL
  let assignmentFile = null;
  if (req.files?.assignmentFile && req.files.assignmentFile.length > 0) {
    const assignmentFileLocalPath = req.files.assignmentFile[0].path;
    assignmentFile = await uploadOnCloudinary(assignmentFileLocalPath);
  }

  const user = await User.findById(req.user._id);

  if (!user) {
    throw new ApiError(404, "User not found.");
  }

  // Find the specific course and subject
  const course = user.courseEnrollments.find(c => c.courseName === courseName);
  if (!course) {
    throw new ApiError(404, "Course not found.");
  }

  const subject = course.subjects.find(s => s.subname === subjectName);
  if (!subject) {
    throw new ApiError(404, "Subject not found.");
  }

  // Find the assignment in the subject
  const assignment = subject.assignments.find(a => a.assignmentName === assignmentName);
  if (!assignment) {
    throw new ApiError(404, "Assignment not found.");
  }

  // Update assignment file and status
  assignment.assignmentFile = assignmentFile.url;
  assignment.status = 'completed';

  // Save the updated user document
  await user.save();

  return res.status(200).json(new ApiResponse(200, {}, "Assignment submitted successfully."));
});


export {
  registerUser,
  loginUser,
  logoutUser,
  refreshAccessToken,
  getUserData,
  submitAssignment,
};


