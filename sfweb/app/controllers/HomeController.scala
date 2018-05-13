package controllers

import javax.inject._
import play.api.libs.json.JsObject
import play.api.libs.json.JsBoolean
import play.api.mvc._

class HomeController @Inject()(cc: ControllerComponents) extends AbstractController(cc) {

  val machineModel = new MachineModel()

  def index = Action {
    Ok(views.html.index())
  }

  def sf = Action { implicit request =>
    // localhost:9000/sf?callType=Fire&prio=1&hood=Portola&dt=01/16/2012 11:11:14 AM -> true
    // localhost:9000/sf?callType=Fire&prio=2&hood=Portola&dt=01/16/2012 11:11:14 AM -> false
    // localhost:9000/sf?callType=Fire&prio=3&hood=Portola&dt=01/16/2012 11:11:14 AM -> true

    val callType = request.getQueryString("callType").get
    val priority = request.getQueryString("prio").get
    val hood = request.getQueryString("hood").get
    val dt = request.getQueryString("dt").get

    val result = machineModel.isLifeThreatening(callType, priority, hood, dt)
    val json = JsObject(Seq(
      "lifeThreatening" -> JsBoolean(result)
    ))
    Ok(json)
  }

}
