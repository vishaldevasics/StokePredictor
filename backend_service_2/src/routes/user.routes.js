import { Router } from "express";
import { loginUser, logoutUser, registerUser, refreshAccessToken, getUserData,submitAssignment} from "../controllers/user.controller.js";
import {upload} from "../middlewares/multer.middleware.js"
import { verifyJWT } from "../middlewares/auth.middleware.js";


const router = Router()

router.route("/register").post(
  upload.fields([
    {
      name: "profilePhoto",
      maxCount: 1,
    }
  ]),
  registerUser
)

router.route("/login").post(loginUser)

//secured routes

router.route("/logout").post(verifyJWT, logoutUser)

router.route("/refresh-token").post(refreshAccessToken)


router.route("/uploadReport").post(upload.fields([
  {
    name: "assignmentFile",
    maxCount: 1,
  }
]),verifyJWT,submitAssignment)

router.route("/data").get(verifyJWT,getUserData)

export default router